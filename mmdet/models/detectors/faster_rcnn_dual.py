import warnings

from torch import nn
from mmdet.core import bbox2result
from mmcv.runner import auto_fp16
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNDual(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, backbone,prim = None,
                 neck=None,
                 rpn_head_acid=None,rpn_head_iodine=None,
                 roi_head_acid=None,roi_head_iodine=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterRCNNDual, self).__init__(init_cfg)
        if prim is None:
            self.prim = ['acid', 'iodine']
        elif not isinstance(prim, list):
            self.prim = [prim]
        else:
            self.prim = prim

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head_acid is not None and rpn_head_iodine is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_acid_ = rpn_head_acid.copy()
            rpn_head_iodine_ = rpn_head_acid.copy()
            rpn_head_acid_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_iodine_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head_acid = build_head(rpn_head_acid_)
            self.rpn_head_iodine = build_head(rpn_head_iodine_)

        if roi_head_acid is not None and roi_head_iodine is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head_acid.update(train_cfg=rcnn_train_cfg)
            roi_head_acid.update(test_cfg=test_cfg.rcnn)
            roi_head_acid.pretrained = pretrained
            self.roi_head_acid = build_head(roi_head_acid)

            roi_head_iodine.update(train_cfg=rcnn_train_cfg)
            roi_head_iodine.update(test_cfg=test_cfg.rcnn)
            roi_head_iodine.pretrained = pretrained
            self.roi_head_iodine = build_head(roi_head_iodine)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg



    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        return super(FasterRCNNDual, self).extract_feat(acid_img), super(FasterRCNNDual, self).extract_feat(iodine_img)

    def forward_dummy(self, acid_img, iodine_img):
        raise NotImplementedError

    def forward_train(self,
                      acid_img, iodine_img,
                      img_metas,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      acid_proposals = None, iodine_proposals = None,
                      **kwargs):
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            acid_rpn_losses, acid_proposal_list = self.rpn_head_acid.forward_train(
                acid_feats, img_metas, acid_gt_bboxes, gt_labels = None, gt_bboxes_ignore = acid_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'acid' in self.prim:
                for k, v in acid_rpn_losses.items():
                    losses['acid_' + k] = v
            iodine_rpn_losses, iodine_proposal_list = self.rpn_head_iodine.forward_train(
                iodine_feats, img_metas, iodine_gt_bboxes, gt_labels = None, gt_bboxes_ignore = iodine_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'iodine' in self.prim:
                for k, v in iodine_rpn_losses.items():
                    losses['iodine_' + k] = v
        else:
            acid_proposal_list = acid_proposals
            iodine_proposal_list = iodine_proposals

        roi_acid_losses = self.roi_head_acid.forward_train(
            acid_feats, iodine_feats, img_metas, acid_proposal_list, acid_gt_bboxes, iodine_gt_bboxes, acid_gt_labels,
            iodine_gt_labels, acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None, **kwargs)
        roi_iodine_losses = self.roi_head_iodine.forward_train(
            iodine_feats, acid_feats, img_metas, iodine_proposal_list, iodine_gt_bboxes, acid_gt_bboxes, iodine_gt_labels,
            acid_gt_labels, iodine_gt_bboxes_ignore=None, acid_gt_bboxes_ignore=None, **kwargs)
        for k, v in roi_acid_losses.items():
            if any([name in k for name in self.prim]):
                losses[k] = v
        for k, v in roi_iodine_losses.items():
            if any([name in k for name in self.prim]):
                losses[k] = v

        return losses

    def forward_test(self, acid_imgs, iodine_imgs, img_metas, **kwargs):
        for var, name in [(acid_imgs, 'acid_imgs'), (iodine_imgs, 'iodine_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(acid_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(acid_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(acid_imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(acid_imgs[0], iodine_imgs[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    @auto_fp16(apply_to = ('acid_img', 'iodine_img'))
    def forward(self, acid_img, iodine_img, img_metas, return_loss = True, **kwargs):
        if return_loss:
            return self.forward_train(acid_img, iodine_img, img_metas, **kwargs)
        else:
            return self.forward_test(acid_img, iodine_img, img_metas, **kwargs)

    def simple_test(self, acid_img, iodine_img, img_metas, acid_proposals = None, iodine_proposals = None, rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        if acid_proposals is not None:
            acid_proposal_list = self.rpn_head_acid.simple_test_rpn(acid_feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is not None:   ##not none??????
            iodine_proposal_list = self.rpn_head_iodine.simple_test_rpn(iodine_feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals

        roi_result_acid = self.roi_head_acid.simple_test(acid_feats, iodine_feats, acid_proposal_list, img_metas, rescale = rescale)
        roi_result_iodine = self.roi_head_iodine.simple_test(iodine_feats, acid_feats, iodine_proposal_list, img_metas, rescale=rescale)

        return (roi_result_acid,roi_result_iodine)

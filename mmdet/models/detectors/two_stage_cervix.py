import warnings
from torch import nn

from mmcv.runner import auto_fp16
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class TwoStageCervixDetector(TwoStageDetector):
    """Base class for two-stage detectors for cervix dataset.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck = None,
                 rpn_head = None,
                 roi_head = None,
                 train_cfg = None,
                 test_cfg = None,
                 pretrained = None,
                 init_cfg = None,
                 prim = None,
                 no_shared_modules = None):
        super(TwoStageDetector, self).__init__(init_cfg)
        self.stages = ['acid', 'iodine']

        if prim is None:
            self.prim = ['acid', 'iodine']
        elif not isinstance(prim, list):
            self.prim = [prim]
        else:
            self.prim = prim

        if no_shared_modules is None:
            self.no_shared_modules = []
        else:
            self.no_shared_modules = no_shared_modules

        self.pretrained = pretrained
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = self.build_backbone(backbone)

        if neck is not None:
            self.neck = self.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = self.build_rpn_head(rpn_head)

        if roi_head is not None:
            self.roi_head = self.build_roi_head(roi_head)

    def build_backbone(self, backbone):
        if self.pretrained:
            warnings.warn('DeprecationWarning: self.pretrained is deprecated, '
                          'please use "init_cfg" instead')
        if 'backbone' in self.no_shared_modules:
            if self.stages[0] not in backbone:
                if self.pretrained:
                    backbone.pretrained = self.pretrained
                backbone = {stage: backbone for stage in self.stages}
            else:
                for stage in self.stages:
                    if self.pretrained:
                        backbone[stage].pretrained = self.pretrained
            backbone = nn.ModuleDict({stage: build_backbone(backbone[stage]) for stage in self.stages})
        else:
            if self.pretrained:
                backbone.pretrained = self.pretrained
            backbone = build_backbone(backbone)
        return backbone

    def build_neck(self, neck):
        if 'neck' in self.no_shared_modules:
            if self.stages[0] not in neck:
                neck = {stage: neck for stage in self.stages}
            neck = nn.ModuleDict({stage: build_neck(neck[stage]) for stage in self.stages})
        else:
            neck = build_neck(neck)
        return neck

    def build_rpn_head(self, rpn_head):
        if 'rpn_head' in self.no_shared_modules:
            rpn_train_cfg = self.train_cfg.rpn if self.train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            if self.stages[0] not in rpn_head_:
                rpn_head_.update(train_cfg = rpn_train_cfg, test_cfg = self.test_cfg.rpn)
                rpn_head_ = {stage: rpn_head_ for stage in self.stages}
            else:
                for stage in self.stages:
                    rpn_head_[stage].update(train_cfg = rpn_train_cfg, test_cfg = self.test_cfg.rpn)
            rpn_head = nn.ModuleDict({stage: build_head(rpn_head_[stage]) for stage in self.stages})
        else:
            rpn_train_cfg = self.train_cfg.rpn if self.train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg = rpn_train_cfg, test_cfg = self.test_cfg.rpn)
            rpn_head = build_head(rpn_head_)
        return rpn_head

    def build_roi_head(self, roi_head):
        if 'roi_head' in self.no_shared_modules:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = self.train_cfg.rcnn if self.train_cfg is not None else None
            if self.stages[0] not in roi_head:
                roi_head.update(train_cfg = rcnn_train_cfg)
                roi_head.update(test_cfg = self.test_cfg.rcnn)
                roi_head.pretrained = self.pretrained
                roi_head = {stage: roi_head for stage in self.stages}
            else:
                for stage in self.stages:
                    roi_head[stage].update(train_cfg = rcnn_train_cfg)
                    roi_head[stage].update(test_cfg = self.test_cfg.rcnn)
                    roi_head[stage].pretrained = self.pretrained
            roi_head = nn.ModuleDict({stage: build_head(roi_head[stage]) for stage in self.stages})
        else:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = self.train_cfg.rcnn if self.train_cfg is not None else None
            roi_head.update(train_cfg = rcnn_train_cfg)
            roi_head.update(test_cfg = self.test_cfg.rcnn)
            roi_head.pretrained = self.pretrained
            roi_head = build_head(roi_head)
        return roi_head

    def _extract_feat(self, img, stage = None):
        """Directly extract features from the backbone+neck."""
        if 'backbone' in self.no_shared_modules:
            x = self.backbone[stage](img)
        else:
            x = self.backbone(img)
        if self.with_neck:
            if 'neck' in self.no_shared_modules:
                x = self.neck[stage](x)
            else:
                x = self.neck(x)
        return x

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        return self._extract_feat(acid_img, 'acid'), self._extract_feat(iodine_img, 'iodine')

    def rpn_forward_train(self, stage = None, *args, **kwargs):
        if stage is not None and 'rpn_head' in self.no_shared_modules:
            return self.rpn_head[stage].forward_train(*args, **kwargs)
        else:
            return self.rpn_head.forward_train(*args, **kwargs)

    def rpn_forward_test(self, stage = None, *args, **kwargs):
        if stage is not None and 'rpn_head' in self.no_shared_modules:
            return self.rpn_head[stage].simple_test_rpn(*args, **kwargs)
        else:
            return self.rpn_head.simple_test_rpn(*args, **kwargs)

    def rpn_train(self, losses,
                  acid_feats, iodine_feats,
                  img_metas,
                  acid_gt_bboxes, iodine_gt_bboxes,
                  acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                  acid_proposals = None, iodine_proposals = None):

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            acid_rpn_losses, acid_proposal_list = self.rpn_forward_train('acid', acid_feats, img_metas, acid_gt_bboxes, gt_labels = None,
                                                                         gt_bboxes_ignore = acid_gt_bboxes_ignore,
                                                                         proposal_cfg = proposal_cfg)
            if 'acid' in self.prim:
                for k, v in acid_rpn_losses.items():
                    losses['acid_' + k] = v
            iodine_rpn_losses, iodine_proposal_list = self.rpn_forward_train('iodine', iodine_feats, img_metas, iodine_gt_bboxes,
                                                                             gt_labels = None, gt_bboxes_ignore = iodine_gt_bboxes_ignore,
                                                                             proposal_cfg = proposal_cfg)
            if 'iodine' in self.prim:
                for k, v in iodine_rpn_losses.items():
                    losses['iodine_' + k] = v
        else:
            acid_proposal_list = acid_proposals
            iodine_proposal_list = iodine_proposals
        return acid_proposal_list, iodine_proposal_list

    def rpn_test(self, acid_feats, iodine_feats, img_metas, acid_proposals = None, iodine_proposals = None):
        if acid_proposals is None:
            acid_proposal_list = self.rpn_forward_test('acid', acid_feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is None:
            iodine_proposal_list = self.rpn_forward_test('iodine', iodine_feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals
        return acid_proposal_list, iodine_proposal_list

    def roi_forward_train(self, stage = None, *args, **kwargs):
        if stage is not None and 'roi_head' in self.no_shared_modules:
            return self.roi_head[stage].forward_train(*args, **kwargs)
        else:
            return self.roi_head.forward_train(*args, **kwargs)

    def roi_forward_test(self, stage = None, *args, **kwargs):
        if stage is not None and 'roi_head' in self.no_shared_modules:
            return self.roi_head[stage].simple_test(*args, **kwargs)
        else:
            return self.roi_head.simple_test(*args, **kwargs)

    def roi_train(self, losses,
                  acid_feats, iodine_feats,
                  img_metas,
                  acid_gt_bboxes, iodine_gt_bboxes,
                  acid_gt_labels, iodine_gt_labels,
                  acid_proposal_list, iodine_proposal_list,
                  acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                  **kwargs):

        acid_roi_losses = self.roi_forward_train('acid', acid_feats, img_metas, acid_proposal_list, acid_gt_bboxes, acid_gt_labels,
                                                 acid_gt_bboxes_ignore, **kwargs)
        if 'acid' in self.prim:
            for k, v in acid_roi_losses.items():
                losses['acid_' + k] = v

        iodine_roi_losses = self.roi_forward_train('iodine', iodine_feats, img_metas, iodine_proposal_list, iodine_gt_bboxes,
                                                   iodine_gt_labels, iodine_gt_bboxes_ignore, **kwargs)
        if 'iodine' in self.prim:
            for k, v in iodine_roi_losses.items():
                losses['iodine_' + k] = v

    def roi_test(self, acid_feats, iodine_feats, img_metas, acid_proposal_list, iodine_proposal_list, rescale = False):
        acid_results = self.roi_forward_test('acid', acid_feats, acid_proposal_list, img_metas, rescale = rescale)
        iodine_results = self.roi_forward_test('iodine', iodine_feats, iodine_proposal_list, img_metas, rescale = rescale)
        return acid_results, iodine_results

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
        acid_proposal_list, iodine_proposal_list = self.rpn_train(losses,
                                                                  acid_feats, iodine_feats,
                                                                  img_metas,
                                                                  acid_gt_bboxes, iodine_gt_bboxes,
                                                                  acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore,
                                                                  acid_proposals, iodine_proposals)
        self.roi_train(losses,
                       acid_feats, iodine_feats,
                       img_metas,
                       acid_gt_bboxes, iodine_gt_bboxes,
                       acid_gt_labels, iodine_gt_labels,
                       acid_proposal_list, iodine_proposal_list,
                       acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore,
                       **kwargs)
        return losses

    def simple_test(self, acid_img, iodine_img, img_metas, acid_proposals = None, iodine_proposals = None, rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        acid_proposal_list, iodine_proposal_list = self.rpn_test(acid_feats, iodine_feats, img_metas, acid_proposals, iodine_proposals)
        acid_results, iodine_results = self.roi_test(acid_feats, iodine_feats, img_metas, acid_proposal_list, iodine_proposal_list, rescale)
        return acid_results, iodine_results

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

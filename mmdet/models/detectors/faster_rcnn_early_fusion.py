import warnings
import numpy as np
import torch
from torch import nn

from mmcv.runner import auto_fp16
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNEarlyFusion(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 prim,
                 rpn_head = None,
                 roi_head = None,
                 pretrained = None,
                 *args, **kwargs):
        super(FasterRCNNEarlyFusion, self).__init__(rpn_head = None, roi_head = None, pretrained = pretrained, *args, **kwargs)
        self.stages = ['acid', 'iodine']

        if prim is None:
            self.prim = ['acid', 'iodine']
        elif not isinstance(prim, list):
            self.prim = [prim]
        else:
            self.prim = prim

        if rpn_head is not None:
            rpn_train_cfg = self.train_cfg.rpn if self.train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            if self.stages[0] not in rpn_head_:
                rpn_head_.update(train_cfg = rpn_train_cfg, test_cfg = self.test_cfg.rpn)
                rpn_head_ = {stage: rpn_head_ for stage in self.stages}
            else:
                for stage in self.stages:
                    rpn_head_[stage].update(train_cfg = rpn_train_cfg, test_cfg = self.test_cfg.rpn)
            self.rpn_head = nn.ModuleDict({stage: build_head(rpn_head_[stage]) for stage in self.stages})

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = self.train_cfg.rcnn if self.train_cfg is not None else None
            if self.stages[0] not in roi_head:
                roi_head.update(train_cfg = rcnn_train_cfg)
                roi_head.update(test_cfg = self.test_cfg.rcnn)
                roi_head.pretrained = pretrained
                roi_head = {stage: roi_head for stage in self.stages}
            else:
                for stage in self.stages:
                    roi_head[stage].update(train_cfg = rcnn_train_cfg)
                    roi_head[stage].update(test_cfg = self.test_cfg.rcnn)
                    roi_head[stage].pretrained = pretrained
            self.roi_head = nn.ModuleDict({stage: build_head(roi_head[stage]) for stage in self.stages})

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
        feats = self.extract_feat(torch.cat([acid_img, iodine_img], dim = 1))

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            acid_rpn_losses, acid_proposal_list = self.rpn_head['acid'].forward_train(
                feats, img_metas, acid_gt_bboxes, gt_labels = None, gt_bboxes_ignore = acid_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'acid' in self.prim:
                for k, v in acid_rpn_losses.items():
                    losses['acid_' + k] = v
            iodine_rpn_losses, iodine_proposal_list = self.rpn_head['iodine'].forward_train(
                feats, img_metas, iodine_gt_bboxes, gt_labels = None, gt_bboxes_ignore = iodine_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'iodine' in self.prim:
                for k, v in iodine_rpn_losses.items():
                    losses['iodine_' + k] = v
        else:
            acid_proposal_list = acid_proposals
            iodine_proposal_list = iodine_proposals

        acid_roi_losses = self.roi_head['acid'].forward_train(feats, img_metas, acid_proposal_list, acid_gt_bboxes, acid_gt_labels,
                                                              acid_gt_bboxes_ignore, **kwargs)
        if 'acid' in self.prim:
            for k, v in acid_roi_losses.items():
                losses['acid_' + k] = v

        iodine_roi_losses = self.roi_head['iodine'].forward_train(feats, img_metas, iodine_proposal_list, iodine_gt_bboxes,
                                                                  iodine_gt_labels,
                                                                  iodine_gt_bboxes_ignore, **kwargs)
        if 'iodine' in self.prim:
            for k, v in iodine_roi_losses.items():
                losses['iodine_' + k] = v

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
        feats = self.extract_feat(torch.cat([acid_img, iodine_img], dim = 1))
        if acid_proposals is None:
            acid_proposal_list = self.rpn_head['acid'].simple_test_rpn(feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is None:
            iodine_proposal_list = self.rpn_head['iodine'].simple_test_rpn(feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals

        acid_results = self.roi_head['acid'].simple_test(feats, acid_proposal_list, img_metas, rescale = rescale)
        iodine_results = self.roi_head['iodine'].simple_test(feats, iodine_proposal_list, img_metas, rescale = rescale)

        return acid_results, iodine_results

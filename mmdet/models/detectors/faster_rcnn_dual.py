import warnings

from torch import nn
from mmdet.core import bbox2result
from mmcv.runner import auto_fp16
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage_cervix import TwoStageCervixDetector


@DETECTORS.register_module()
class FasterRCNNDual(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""
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
            acid_rpn_losses, acid_proposal_list = self.rpn_head.forward_train(
                acid_feats, img_metas, acid_gt_bboxes, gt_labels = None, gt_bboxes_ignore = acid_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'acid' in self.prim:
                for k, v in acid_rpn_losses.items():
                    losses['acid_' + k] = v
            iodine_rpn_losses, iodine_proposal_list = self.rpn_head.forward_train(
                iodine_feats, img_metas, iodine_gt_bboxes, gt_labels = None, gt_bboxes_ignore = iodine_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'iodine' in self.prim:
                for k, v in iodine_rpn_losses.items():
                    losses['iodine_' + k] = v
        else:
            acid_proposal_list = acid_proposals
            iodine_proposal_list = iodine_proposals

        roi_losses = self.roi_head.forward_train(
            acid_feats, iodine_feats, img_metas, acid_proposal_list, iodine_proposal_list, acid_gt_bboxes, iodine_gt_bboxes, acid_gt_labels,
            iodine_gt_labels, acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None, **kwargs)
        for k, v in roi_losses.items():
            if any([name in k for name in self.prim]):
                losses[k] = v

        return losses

    def simple_test(self, acid_img, iodine_img, img_metas, acid_proposals = None, iodine_proposals = None, rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        if acid_proposals is None:
            acid_proposal_list = self.rpn_head.simple_test_rpn(acid_feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is None:
            iodine_proposal_list = self.rpn_head.simple_test_rpn(iodine_feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals

        return self.roi_head.simple_test(acid_feats, iodine_feats, acid_proposal_list, iodine_proposal_list, img_metas, rescale = rescale)

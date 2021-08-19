from ..builder import DETECTORS
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
        acid_proposal_list, iodine_proposal_list = self.rpn_train(losses,
                                                                  acid_feats, iodine_feats,
                                                                  img_metas,
                                                                  acid_gt_bboxes, iodine_gt_bboxes,
                                                                  acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore,
                                                                  acid_proposals, iodine_proposals)
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
        acid_proposal_list, iodine_proposal_list = self.rpn_test(acid_feats, iodine_feats, img_metas, acid_proposals, iodine_proposals)
        return self.roi_head.simple_test(acid_feats, iodine_feats, acid_proposal_list, iodine_proposal_list, img_metas, rescale = rescale)

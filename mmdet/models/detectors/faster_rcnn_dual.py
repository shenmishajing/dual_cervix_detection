from ..builder import DETECTORS
from .two_stage_cervix import TwoStageCervixDetector


@DETECTORS.register_module()
class FasterRCNNDual(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def roi_train(self, losses,
                  acid_feats, iodine_feats,
                  img_metas,
                  acid_gt_bboxes, iodine_gt_bboxes,
                  acid_gt_labels, iodine_gt_labels,
                  acid_proposal_list, iodine_proposal_list,
                  acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                  **kwargs):

        roi_losses = self.roi_forward_train(
            None, acid_feats, iodine_feats, img_metas, acid_proposal_list, iodine_proposal_list, acid_gt_bboxes, iodine_gt_bboxes,
            acid_gt_labels, iodine_gt_labels, acid_gt_bboxes_ignore = acid_gt_bboxes_ignore,
            iodine_gt_bboxes_ignore = iodine_gt_bboxes_ignore, **kwargs)
        for k, v in roi_losses.items():
            if any([name in k for name in self.prim]):
                losses[k] = v

    def roi_test(self, acid_feats, iodine_feats, img_metas, acid_proposal_list, iodine_proposal_list, rescale = False):
        return self.roi_forward_test(None, acid_feats, iodine_feats, acid_proposal_list, iodine_proposal_list, img_metas,
                                     rescale = rescale)

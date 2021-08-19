import torch
from torch import nn

from ..builder import DETECTORS
from .two_stage_cervix import TwoStageCervixDetector


@DETECTORS.register_module()
class FasterRCNNMiddleFusion(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, *args, **kwargs):
        super(FasterRCNNMiddleFusion, self).__init__(*args, **kwargs)

        self.fusion_module = nn.ModuleList([
            nn.Conv2d(self.neck[self.stages[0]].out_channels * 2, self.neck[self.stages[0]].out_channels, 3)
            for _ in range(self.neck[self.stages[0]].num_outs)
        ])

    def feature_fusion(self, acid_feats, iodine_feats):
        feats = []
        for i in range(len(self.fusion_module)):
            acid_feat = acid_feats[i]
            iodine_feat = iodine_feats[i]
            feat = torch.cat([acid_feat, iodine_feat], dim = 1)
            feat = self.fusion_module[i](feat)
            feats.append(feat)
        return feats

    def forward_train(self,
                      acid_img, iodine_img,
                      img_metas,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      acid_proposals = None, iodine_proposals = None,
                      **kwargs):
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        feats = self.feature_fusion(acid_feats, iodine_feats)

        losses = dict()
        acid_proposal_list, iodine_proposal_list = self.rpn_train(losses,
                                                                  feats, feats,
                                                                  img_metas,
                                                                  acid_gt_bboxes, iodine_gt_bboxes,
                                                                  acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore,
                                                                  acid_proposals, iodine_proposals)
        self.roi_train(losses,
                       feats, feats,
                       img_metas,
                       acid_gt_bboxes, iodine_gt_bboxes,
                       acid_gt_labels, iodine_gt_labels,
                       acid_proposal_list, iodine_proposal_list,
                       acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore,
                       **kwargs)

        return losses

    def simple_test(self, acid_img, iodine_img, img_metas, acid_proposals = None, iodine_proposals = None, rescale = False):
        """Test without augmentation."""
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        feats = self.feature_fusion(acid_feats, iodine_feats)
        acid_proposal_list, iodine_proposal_list = self.rpn_test(feats, feats, img_metas, acid_proposals, iodine_proposals)
        acid_results, iodine_results = self.roi_test(feats, feats, img_metas, acid_proposal_list, iodine_proposal_list, rescale)
        return acid_results, iodine_results

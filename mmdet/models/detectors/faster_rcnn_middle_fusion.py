import torch
from torch import nn

from ..builder import DETECTORS
from .two_stage_cervix import TwoStageCervixDetector


@DETECTORS.register_module()
class FasterRCNNMiddleFusion(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, *args, **kwargs):
        super(FasterRCNNMiddleFusion, self).__init__(*args, **kwargs)

        if isinstance(self.neck, nn.ModuleDict):
            self.fusion_module_in_channels = self.neck[self.stages[0]].out_channels
            self.fusion_module_num = self.neck[self.stages[0]].num_outs
        else:
            self.fusion_module_in_channels = self.neck.out_channels
            self.fusion_module_num = self.neck.num_outs

        self.fusion_module = nn.ModuleList([
            nn.Conv2d(self.fusion_module_in_channels * 2, self.fusion_module_in_channels, 1)
            for _ in range(self.fusion_module_num)
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

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        acid_feats, iodine_feats = super(FasterRCNNMiddleFusion, self).extract_feat(acid_img, iodine_img)
        feats = self.feature_fusion(acid_feats, iodine_feats)
        return feats, feats
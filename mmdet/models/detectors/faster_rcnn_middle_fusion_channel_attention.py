import torch
from torch import nn

from ..builder import DETECTORS
from .faster_rcnn_middle_fusion import FasterRCNNMiddleFusion


@DETECTORS.register_module()
class FasterRCNNMiddleFusionChannelAttention(FasterRCNNMiddleFusion):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, pool_size = 1, *args, **kwargs):
        super(FasterRCNNMiddleFusionChannelAttention, self).__init__(*args, **kwargs)

        pool_area = pool_size ** 2
        middle_channels = 256
        self.max_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.attn_module = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.fusion_module_in_channels * 2 * pool_area, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, self.fusion_module_in_channels * 2 * pool_area),
                nn.Sigmoid()
            ])
            for _ in range(self.fusion_module_num)
        ])

    def feature_fusion(self, acid_feats, iodine_feats):
        feats = []
        for i in range(len(self.attn_module)):
            acid_feat = acid_feats[i]
            iodine_feat = iodine_feats[i]
            feat = torch.cat([acid_feat, iodine_feat], dim = 1)
            attn = self.max_pool(feat) + self.avg_pool(feat)
            attn = attn.reshape(attn.shape[0], -1)
            for m in self.attn_module[i]:
                attn = m(attn)
            feat = feat * attn[..., None, None]
            feat = self.fusion_module[i](feat)
            feats.append(feat)
        return feats

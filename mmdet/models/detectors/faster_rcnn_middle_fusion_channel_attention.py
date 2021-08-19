import torch
from torch import nn

from ..builder import DETECTORS
from .faster_rcnn_middle_fusion import FasterRCNNMiddleFusion


@DETECTORS.register_module()
class FasterRCNNMiddleFusionChannelAttention(FasterRCNNMiddleFusion):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, pool_size = 1, *args, **kwargs):
        super(FasterRCNNMiddleFusion, self).__init__(*args, **kwargs)

        pool_area = pool_size ** 2
        middle_channels = 256
        self.max_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fusion_module = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.neck[self.stages[0]].out_channels * 2 * pool_area, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, middle_channels),
                nn.ReLU(),
                nn.Linear(middle_channels, self.neck[self.stages[0]].out_channels * 2 * pool_area),
                nn.Sigmoid()
            ])
            for _ in range(self.neck[self.stages[0]].num_outs)
        ])
        self.down_channel_module = nn.ModuleList([
            nn.Conv2d(self.neck[self.stages[0]].out_channels * 2, self.neck[self.stages[0]].out_channels, 1)
            for _ in range(self.neck[self.stages[0]].num_outs)
        ])

    def feature_fusion(self, acid_feats, iodine_feats):
        feats = []
        for i in range(len(self.fusion_module)):
            acid_feat = acid_feats[i]
            iodine_feat = iodine_feats[i]
            feat = torch.cat([acid_feat, iodine_feat], dim = 1)
            attn = self.max_pool(feat) + self.avg_pool(feat)
            attn = attn.reshape(attn.shape[0], -1)
            for m in self.fusion_module[i]:
                attn = m(attn)
            feat = feat * attn[..., None, None]
            feat = self.down_channel_module[i](feat)
            feats.append(feat)
        return feats

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import (PointGenerator, build_assigner,
                        build_sampler, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class SparseRepPoints(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 num_points=9,
                 top_k=100,
                 stacked_linears=3,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 **kwargs):

        self.stacked_linears = stacked_linears

        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.top_k = top_k
        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        # - dcn_base_y = array([-1, -1, -1,  0,  0,  0,  1,  1,  1])
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        # - dcn_base_x = array([-1,  0,  1, -1,  0,  1, -1,  0,  1])
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack(
            [dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        # - dcn_base_offset.shape = [1, 18, 1, 1] 18: (-1, -1), (-1, 0)...
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        pass

    def _init_layers(self):
        """Initialize layers of the head."""

        self.convs = nn.ModuleList()
        for i in range(self.stacked_convs):  # - stacked_convs = 3
            chn = self.in_channels if i == 0 else self.feat_channels
            self.convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        pts_out_dim = 2 * self.num_points
        self.offset1_conv = nn.Conv2d(self.feat_channels,
                                      self.point_feat_channels,
                                      3, 1, 1)
        self.offset1_out = nn.Conv2d(self.point_feat_channels,
                                     pts_out_dim,
                                     1, 1, 0)

        self.objectness_conv = nn.Conv2d(self.feat_channels,
                                         self.feat_channels,
                                         3, 1, 1)
        self.objectness_out = nn.Conv2d(self.point_feat_channels,
                                        1,
                                        1, 1, 0)
        # self.objectness_sigmoid_out = nn.Sigmoid()

        self.encode_points_linears = nn.ModuleList()
        for i in range(self.stacked_linears):
            units = pts_out_dim if i == 0 else self.point_feat_channels
            self.encode_points_linears.append(
                nn.Linear(units, self.point_feat_channels))

        concat_feat_channels = (self.num_points + 1) * self.point_feat_channels
        self.concat_feat_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = concat_feat_channels if (i + 1 < self.stacked_convs) else self.point_feat_channels
            self.concat_feat_convs.append(
                nn.Conv1d(concat_feat_channels,
                          chn,
                          3, 1, 1))

        self.reg_conv = nn.Conv1d(concat_feat_channels,
                                  concat_feat_channels // 2,
                                  3, 1, 1)
        self.reg_out = nn.Conv1d(concat_feat_channels // 2,
                                 4,
                                 1, 1, 0)

        self.cls_conv = nn.Conv1d(concat_feat_channels,
                                  concat_feat_channels // 2,
                                  3, 1, 1)
        self.cls_out = nn.Conv1d(concat_feat_channels // 2,
                                 self.num_classes,
                                 1, 1, 0)
        

    def init_weights(self):
        for m in self.convs:
            normal_init(m.conv, std=0.01)

        normal_init(self.offset1_conv, std=0.01)
        normal_init(self.offset1_out, std=0.01)

        normal_init(self.objectness_conv, std=0.01)
        normal_init(self.objectness_out, std=0.01)

        for m in self.encode_points_linears:
            normal_init(m, std=0.01)
        
        for m in self.concat_feat_convs:
            normal_init(m.conv, std=0.01)
        
        normal_init(self.reg_conv, std=0.01)
        normal_init(self.reg_out, std=0.01)

        normal_init(self.cls_conv, std=0.01)
        normal_init(self.cls_out, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        # - Forward feature map of a single FPN level.

        pass

    def get_targets(self):

        pass

    def loss_single(self):

        pass

    def loss(self):

        pass

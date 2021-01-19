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
class SparseRepPointsHead(AnchorFreeHead):

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
        # - dcn_base_offset.shape = [1, 18, 1] 18: (-1, -1), (-1, 0)...
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1)

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        pass

    def _init_layers(self):
        """Initialize layers of the head."""

        self.relu = nn.ReLU(inplace=True)

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
        self.offset_conv = nn.Conv2d(self.feat_channels,
                                      self.point_feat_channels,
                                      3, 1, 1)
        self.offset_out = nn.Conv2d(self.point_feat_channels,
                                     pts_out_dim,
                                     1, 1, 0)

        self.objectness_conv = nn.Conv2d(self.feat_channels,
                                         self.feat_channels,
                                         3, 1, 1)
        self.objectness_out = nn.Conv2d(self.point_feat_channels,
                                        1,
                                        1, 1, 0)
        self.objectness_sigmoid_out = nn.Sigmoid()

        self.encode_points_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = pts_out_dim if i == 0 else self.point_feat_channels
            self.encode_points_convs.append(
                nn.Conv1d(chn,
                          self.point_feat_channels,
                          3, 1, 1))

        concat_feat_channels = (self.num_points + 1) * self.point_feat_channels
        self.concat_feat_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.concat_feat_convs.append(
                nn.Conv1d(concat_feat_channels,
                          concat_feat_channels,
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
                                 self.num_classes + 1,
                                 1, 1, 0)
        
    def init_weights(self):
        for m in self.convs:
            normal_init(m.conv, std=0.01)

        normal_init(self.offset_conv, std=0.01)
        normal_init(self.offset_out, std=0.01)

        normal_init(self.objectness_conv, std=0.01)
        normal_init(self.objectness_out, std=0.01)

        for m in self.encode_points_convs:
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
        # - dcn_base_offset.shape = [1, 18, 1] 18: (-1, -1), (-1, 0)...
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        feat = x 
        for conv in self.convs:
            feat = conv(feat)

        offset = self.offset_out(
            self.relu(self.offset_conv(feat)))
        objectness = self.objectness_sigmoid_out(
            self.objectness_out(self.objectness_conv(feat)))
        # - topn_objectness shape = [B, 1, topn], top_idx shape = [B, 1, topn]
        topn_objectness, topn_idx = torch.topk(objectness.view((objectness.shape[0], 1, -1)), self.top_k, dim=-1)
        w = objectness.shape[-1]
        # - topn_grid_coord shape = [B, 2, topn] 
        topn_grid_coord = torch.stack([topn_idx[..., -2] / w, topn_idx[..., -1] % w], axis=1)
        # - topn_offset shape = [B, 2 * num_points, topn]
        topn_offset = torch.gather(offset.view((offset.shape[0], 2 * self.num_points, -1)), -1, topn_idx)
        topn_points = topn_offset + topn_grid_coord.repeat((1, self.num_points, 1))
        topn_points = topn_points / w 

        topn_points_transposed = topn_points.view((-1, self.num_points, 2, self.top_k)).transpose(-1, -2)
        topn_feat = torch.nn.functional.grid_sample(feat, topn_points_transposed)
        topn_feat = topn_feat.view((topn_feat.shape[0] , self.num_points * self.point_feat_channels, -1))

        topn_points_encoded = topn_points
        for conv in self.encode_points_convs:
            topn_points_encoded = conv(topn_points_encoded)

        topn_feat_concat = torch.cat([topn_feat, topn_points_encoded], axis=-2)
        for conv in self.concat_feat_convs:
            topn_feat_concat = conv(topn_feat_concat)
        
        topn_box = self.reg_out(self.reg_conv(topn_feat_concat))
        topn_cls = self.cls_out(self.cls_conv(topn_feat_concat))

        return objectness, topn_box, topn_cls

    def get_targets(self):

        pass

    def loss_single(self):

        pass

    def loss(self):

        pass

    def get_bboxes(self):

        pass
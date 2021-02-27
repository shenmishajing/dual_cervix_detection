import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import (build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


class RefineHead(object):
    """
    FPN的不同输出，权重复用
    不同迭代次数是否权重复用，不知道
    """
    def __init__(self,
                 cls_out_channels,
                 in_channels=256,
                 feat_channels=256,
                 point_feat_channels=256,
                 num_points=9,
                 topn=10,
                 stacked_linears=2,
                 stacked_encode=2,
                 output_strides=[8,16,32,64,128],
                 loss_bbox=dict(type='GIoULoss'),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                 ):
        self.cls_out_channels = cls_out_channels
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.num_points = num_points
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        # - dcn_base_offset.shape = [1, 18, 1] 18: (-1, -1), (-1, 0)...
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.topn = topn 
        self.stacked_linears = stacked_linears
        self.stacked_encode = stacked_encode
        self.output_strides = output_strides
        
        self.bbox_weights = [1.0, 1.0, 2.0, 2.0]

        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)


    def init_layers(self):
        # 可形变卷积
        self.dcn = DeformConv2d(self.in_channels, 
                                self.feat_channels,
                                self.dcn_kernel, 1,
                                self.dcn_pad).cuda()
        
        self.relu = nn.ReLU(inplace=True).cuda()
        
        pts_out_dim = 2 * self.num_points
        self.offset_conv = nn.Conv2d(self.feat_channels,
                                     self.point_feat_channels,
                                     3, 1, 1).cuda()
        self.offset_out = nn.Conv2d(self.point_feat_channels,
                                    pts_out_dim,
                                    1, 1, 0).cuda()

        # points encoding
        self.encode_points_linear = nn.Linear(pts_out_dim, self.point_feat_channels, False).cuda()
        
        concat_feat_channels = (self.num_points + 1) * self.point_feat_channels
        self.concat_feat_linears = nn.ModuleList([
            nn.Linear(concat_feat_channels, self.feat_channels, False).cuda(),
            nn.ReLU(inplace=True).cuda(),
            nn.Linear(self.feat_channels, self.feat_channels, False).cuda(),
            nn.ReLU(inplace=True).cuda()
        ])

        self.reg_linear = nn.Linear(self.feat_channels, self.feat_channels).cuda()
        self.reg_out = nn.Linear(self.feat_channels, 4).cuda()

        self.cls_out = nn.Linear(self.feat_channels, self.cls_out_channels).cuda()


    def init_weights(self):
        normal_init(self.dcn, std=0.01)

        normal_init(self.offset_conv, std=0.01)
        normal_init(self.offset_out, std=0.01)

        normal_init(self.encode_points_linear, std=0.01)

        for m in self.concat_feat_linears:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
        
        normal_init(self.reg_linear, std=0.01)
        normal_init(self.reg_out, std=0.01)
        
        normal_init(self.cls_out, std=0.01)


    def forward_single(self, feat, dcn_offset, topn_idx, topn_points, prev_topn_bbox):
        """

            Args:
                feat (tensor): shape = [B, c_in, feat_h, feat_w]
                dcn_offset (tensor): shape = [B, num_points * 2, feat_h, feat_w]
                topn_idx (tensor): shape = [B, 1, topn] 
                topn_points (tensor): shape = [B, num_points * 2, topn]
                    ! unormalized
                prev_topn_bbox (tensor): shape = [B, topn, 4]
            
            Returns:
                topn_box_refine (tensor): shape = [B, topn, 4]
                topn_cls_refine (tensor): shape = [B, topn, num_classes]
                offset_refine (tensor): shape = [B, num_points * 2, topn]
                topn_points_refine_unormalized (tensor): shape = [B, topn, 4]
        """
        batch_size = feat.shape[0]
        feat_h, feat_w = feat.shape[-2:]
        dcn_base_offset = self.dcn_base_offset.type_as(feat)
        dcn_offset = dcn_offset - dcn_base_offset
        # dcn_feat shape = feat shape
        dcn_feat = self.dcn(feat, dcn_offset)

        # offset shape = [B, num_points * 2, feat_h, feat_w]
        offset = self.offset_out(self.relu(self.offset_conv(dcn_feat)))
        topn_idx_repeat = topn_idx.repeat([1, 2 * self.num_points, 1])
        # topn_offset shape = [B, num_points * 2, topn]
        topn_offset = torch.gather(offset.view(batch_size, 2 * self.num_points, -1), -1, topn_idx_repeat) 
        
        topn_points_refine = topn_offset + topn_points
        topn_points_refine_normalized = topn_points_refine
        factor = torch.tensor([feat_h, feat_w], dtype=torch.float32).repeat([self.num_points,]).view((1, 2 * self.num_points, 1)).cuda()        
        topn_points_refine_normalized = topn_points_refine_normalized / factor
        topn_points_refine_normalized = 2.0 * topn_points_refine_normalized - 1.0
        # topn_points_refine_transposed shape = [B, num_points, topn, 2]
        topn_points_refine_transposed = topn_points_refine_normalized.view((-1, self.num_points, 2, self.topn)).transpose(-1, -2)     
        # top_feat shape = [B, num_points, topn, c]
        topn_feat = torch.nn.functional.grid_sample(dcn_feat, topn_points_refine_transposed)
        # top_feat shape = [B, num_points * point_feat_channels, topn]
        topn_feat = topn_feat.view((batch_size, self.num_points * self.point_feat_channels, -1)).transpose(-2,-1)

        # topn_points_refine_encoded shape = [B, num_points * 2, topn]
        topn_points_refine_encoded = torch.transpose(topn_points_refine_normalized, -2, -1)
        # topn_points_refine_encoded shape = [B, point_feat_channels, topn]
        topn_points_refine_encoded = self.encode_points_linear(topn_points_refine_encoded)
        
        # topn_feat_concat shape = [B, (num_points + 1) * point_feat_channels, topn]
        topn_feat_concat = torch.cat([topn_feat, topn_points_refine_encoded], axis=-1)
        # topn_feat_concat shape = [B, (num_points + 1) * point_feat_channels, topn]
        for l in self.concat_feat_linears:
            topn_feat_concat = l(topn_feat_concat)

        # topn_feat_concat shape = [B, topn, (num_points + 1) * point_feat_channels]
        topn_box_delta = self.reg_out(self.reg_linear(topn_feat_concat))
        topn_box_refine = self.apply_delta(topn_box_delta, prev_topn_bbox)
        topn_cls_refine = self.cls_out(topn_feat_concat)

        return topn_box_refine, topn_cls_refine, offset, topn_points_refine   


    def apply_delta(self, topn_delta, topn_boxes):
        """
            Sparse RCNN 中的坐标变换方式
            Args:
                topn_delta (tensor): shape = [B, topn, 4], 4: (cx_delta, cy_delta, w_delta, h_delta)
                topn_boxes (tensor): shape = [B, tpon, 4], 4: (cx, cy, w, h)
            
            Returns:
                pred_boxes (tensor): shape = [B, topn, 4], 4: (cx, cy, w, h) 
        """
        ctr_x = topn_boxes[..., 0:1]
        ctr_y = topn_boxes[..., 1:2]
        widths = topn_boxes[..., 2:3]
        heights = topn_boxes[..., 3:4]

        wx, wy, ww, wh = self.bbox_weights
        dx = topn_delta[..., 0:1] / wx
        dy = topn_delta[..., 1:2] / wy
        dw = topn_delta[..., 2:3] / ww
        dh = topn_delta[..., 3:4] / wh

        # Prevent sending too large values into torch.exp()
        # dw = torch.clamp(dw, max=self.scale_clamp)
        # dh = torch.clamp(dh, max=self.scale_clamp)
        
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.cat([pred_ctr_x, pred_ctr_y, pred_w, pred_h], dim=-1)

        return pred_boxes


    def loss_single(self, bbox_refine,
                    cls_refine,
                    bbox_gt, bbox_weights,
                    cls_gt, cls_weights,
                    num_total_samples):
        """
            在不同scale下进行
            
            Args:
                bbox_refine (tensor): shape = [B * topn, 4], 4: (cx, cy, w, h)
                cls_refine (tensor): shape = [B * topn, num_classes]
                bbox_gt (tensor): shape = [B * topn, 4], 4: (cx, cy, w, h)
                bbox_weights (tensor): shape = [B * topn, 4]
                cls_gt (tensor): shape = [B * topn, num_classes]
                cls_weights (tensor): shape = [B * topn]
                num_total_samples (int): 

            Returns:
                loss_bbox, loss_cls 
        """
        
        loss_bbox = self.loss_bbox(bbox_refine,
                                   bbox_gt,
                                   bbox_weights,
                                   avg_factor=num_total_samples)

        loss_cls = self.loss_cls(cls_refine,
                                 cls_gt,
                                 cls_weights, 
                                 avg_factor=num_total_samples)

        return loss_bbox, loss_cls

@HEADS.register_module()
class SparseRepPointsHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 num_points=9,
                 topn=10,
                 stacked_convs=3,
                 stacked_linears=2,
                 stacked_encode=2,
                 output_strides=[8, 16, 32, 64, 128],
                 loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True),
                 loss_bbox=dict(type='GIoULoss'),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                 train_cfg=dict(
                     assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.4,
                        min_pos_iou=0,
                        ignore_iof_thr=-1)),
                 refine_times=0,
                 **kwargs):

        self.stacked_linears = stacked_linears
        self.stacked_encode = stacked_encode
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.topn = topn

        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        # - dcn_base_offset.shape = [1, 18, 1] 18: (-1, -1), (-1, 0)...
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1)

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.output_strides = output_strides
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        #TODO debug
        self.train_cfg = train_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg["assigner"])
            # # - use PseudoSampler when sampling is False
            # if self.sampling and hasattr(self.train_cfg, 'sampler'):
            #     sampler_cfg = self.train_cfg["sampler"]
            # else:
            #     #- A pseudo sampler that does not do sampling actually.Directly returns the positive and negative indices  of samples.
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg)
        
        self.loss_obj = build_loss(loss_obj)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        
        self.refine_times = refine_times
        if refine_times > 0:
            self.refineHead_list =[RefineHead(self.cls_out_channels,
                                              self.in_channels,
                                              self.feat_channels, 
                                              self.point_feat_channels,
                                              self.num_points, 
                                              self.topn, 
                                              self.stacked_linears, 
                                              self.stacked_encode, 
                                              self.output_strides, 
                                              loss_bbox, 
                                              loss_cls) for _ in range(self.refine_times)]
            for h in self.refineHead_list:
                h.init_layers()
                h.init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""

        self.relu = nn.ReLU(inplace=True)
        pts_out_dim = 2 * self.num_points
        self.offset_conv = nn.Conv2d(self.feat_channels,
                                     self.point_feat_channels,
                                     3, 1, 1)
        self.offset_out = nn.Conv2d(self.point_feat_channels,
                                     pts_out_dim,
                                     1, 1, 0)

        self.objectness_conv = nn.Conv2d(self.feat_channels,
                                         self.point_feat_channels,
                                         3, 1, 1)
        self.objectness_out = nn.Conv2d(self.point_feat_channels,
                                        1,
                                        1, 1, 0)
        self.objectness_sigmoid_out = nn.Sigmoid()

        self.encode_points_linear = nn.Linear(pts_out_dim, self.point_feat_channels, False)

        concat_feat_channels = (self.num_points + 1) * self.point_feat_channels
        self.concat_feat_linears = nn.ModuleList([
            nn.Linear(concat_feat_channels, self.feat_channels, False),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_channels, self.feat_channels, False),
            nn.ReLU(inplace=True)
        ])

        # transposed [B, (k*k+1)*C, topn] -> [B, topn, (k*k+1) *C]
        self.reg_linear = nn.Linear(self.feat_channels, self.feat_channels)
        self.reg_sigmoid_out = nn.Sigmoid()
        self.reg_out = nn.Linear(self.feat_channels, 4)

        self.cls_out = nn.Linear(self.feat_channels, self.cls_out_channels)

    def init_weights(self):
        normal_init(self.offset_conv, std=0.01)
        normal_init(self.offset_out, std=0.01)

        normal_init(self.objectness_conv, std=0.01)
        normal_init(self.objectness_out, std=0.01)

        normal_init(self.encode_points_linear, std=0.01)
        
        for m in self.concat_feat_linears:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
        
        normal_init(self.reg_linear, std=0.01)
        normal_init(self.reg_out, std=0.01)

        normal_init(self.cls_out, std=0.01)
    
    def forward(self, feats):
        """
            Returns: 
                objectness_logits (list[tensor]): [obj_scale1, obj_scale2, ..., obj_scale5]
                    obj_scale1: shape = [B, 1, H, W] 
                topn_box (list[tensor]): [topn_box_scale1, topn_box_scale2, ..., topn_box_scale5]
                    topn_box_scale1: shape = [B. topn, 4]
                topn_cls (list[tensor]): [topn_cls_scale1, topn_cls_scale2, ..., topn_cls_scale5]
                    topn_cls_scale1: shape = [B, topn, num_classes]
                ! refine_times > 0，返回 topn_box_refine_list, topn_cls_refine_list
                topn_box_refine_list (list[list[tensor]]): [
                    [topn_box_refine1_scale1, topn_box_refine2_scale1, ..., topn_box_refinen_scale1],
                    [topn_box_refine1_scale2, topn_box_refine2_scale2, ..., topn_box_refinen_scale2],
                    [], ...]
                    topn_box_refine1_scale1: shape = [B, topn, 4]
                topn_cls_refine_list: [
                    [topn_cls_refine1_scale1, topn_cls_refine2_scale1, ..., topn_cls_refinen_scale1],
                    [topn_cls_refine1_scale2, topn_cls_refine2_scale2, ..., topn_cls_refinen_scale2],
                    [], ...]
                    topn_cls_refine1_scale1: shape = [B, topn, num_classes]
                topn_idx (list[tensor]): [topn_idx_scale1, topn_idx_scale2, ..., topn_idx_scale5]
                    topn_idx_scale1: shape = [B, topn, 1]
        """
        result =  multi_apply(self.forward_single, feats)
        return result

    def forward_single(self, feat):
        # - Forward feature map of a single FPN level.
        batch_size = feat.shape[0]
        feat_h, feat_w = feat.shape[-2:]

        # - offset shape = [B, feat_channels, feat_h, feat_w]
        offset = self.offset_out(
            self.relu(self.offset_conv(feat)))

        # - objectness shape = [B, 1, feat_h, feat_w]
        objectness_logits = self.objectness_out(self.objectness_conv(feat))
        objectness = self.objectness_sigmoid_out(objectness_logits)

        # - topn_objectness shape = [B, 1, topn], top_idx shape = [B, 1, topn]
        topn_objectness, topn_idx = torch.topk(objectness.view((batch_size, 1, feat_h * feat_w)), self.topn, dim=-1)
        # - topn_grid_coord shape = [B, 2, topn] topn_idx中的索引为[0, H * W - 1],这里转化为二维索引
        topn_grid_coord = torch.cat([topn_idx // feat_w, topn_idx % feat_w], dim=1)

        # - topn_offset shape = [B, 2 * num_points, topn]
        topn_idx_repeat = topn_idx.repeat([1, 2 * self.num_points, 1])
        topn_offset = torch.gather(offset.view((batch_size, 2 * self.num_points, -1)), -1, topn_idx_repeat)
        # - topn_points shape = [B, 2 * self.num_points, topn]
        topn_points = topn_offset + topn_grid_coord.repeat((1, self.num_points, 1))
        topn_points_normalized = topn_points
        # - factor shape = [1, 2 * num_points, 1]
        factor = torch.tensor([feat_h, feat_w], dtype=torch.float32).repeat([self.num_points,]).view((1, 2 * self.num_points, 1)).cuda()        
        topn_points_normalized = topn_points_normalized / factor
        topn_points_normalized = 2.0 * topn_points_normalized - 1.0
        # - topn_points_transposed shape = [B, num_points, topn, 2] 
        topn_points_transposed = topn_points_normalized.view((-1, self.num_points, 2, self.topn)).transpose(-1, -2)
        
        # print("topn_points_transposed", topn_points_transposed.min(), topn_points_transposed.max(), topn_points_transposed.mean(), topn_points_transposed.std())
        # invalid_offset = torch.logical_or(topn_points_transposed < -1.0, topn_points_transposed > 1.0)
        # num = invalid_offset.sum()
        # s = invalid_offset.numel()
        # print("invalid_offset %d/%d" % (int(num), int(s)))

        # - topn_feat shape = [B, point_feat_channels, num_points, topn] 
        # ! 要求给出索引范围必须（需要标准化）是[-1，1]
        topn_feat = torch.nn.functional.grid_sample(feat, topn_points_transposed)
        # - topn_feat shape = [B, topn, num_points * point_feat_channels] 
        topn_feat = topn_feat.view((batch_size, self.num_points * self.point_feat_channels, self.topn)).transpose(-2, -1)

        # - topn_points_encoded shape = [B, topn, 2 * num_points] 
        topn_points_encoded = torch.transpose(topn_points_normalized, -2, -1)
        topn_points_encoded = self.encode_points_linear(topn_points_encoded)

        # - topn_feat_concat shape = [B, topn, (num_points + 1) * point_feat_channels]
        topn_feat_concat = torch.cat([topn_feat, topn_points_encoded], axis=-1)
        #- topn_feat_concat shape = [B, topn, point_feat_channels]
        for l in self.concat_feat_linears:
            topn_feat_concat = l(topn_feat_concat)

        topn_box  = self.reg_out(self.reg_linear(topn_feat_concat))
        # topn_box = self.reg_sigmoid_out(topn_box)
        topn_cls = self.cls_out(topn_feat_concat)

        # ! 将box的预测理解为(dx, dy, w, h), 那么就不能经过sigmoid，不然dx, dy都是正的量
        # - topn_grid_coord shape = [B, topn, 2], 2: (y, x)
        topn_grid_coord = topn_grid_coord.transpose(-2, -1)
        topn_grid_coord = torch.cat([topn_grid_coord[..., 1:2] / feat_w, topn_grid_coord[..., 0:1] / feat_h], dim=-1)
        topn_box = torch.cat([topn_box[..., :2] + topn_grid_coord, topn_box[..., 2:].sigmoid()], dim=-1)

        if self.refine_times > 0:
            topn_box_refine_list = []
            topn_cls_refine_list = []
            topn_box_refine = topn_box
            topn_points_refine = topn_points
            offset_refine = offset
            for h in self.refineHead_list:
                (topn_box_refine,   
                 topn_cls_refine,
                 tmp_offset,
                 topn_points_refine)  = h.forward_single(x,
                                                         offset_refine,
                                                         topn_idx,
                                                         topn_points_refine,
                                                         topn_box_refine)
                offset_refine = offset_refine + tmp_offset
                topn_box_refine_list.append(topn_box_refine)
                topn_cls_refine_list.append(topn_cls_refine)

        # - topn_idx shape = [B, topn, 1]
        topn_idx = torch.transpose(topn_idx, -2, -1)

        #! objectness看作是分类时，损失函数要求返回值需为logits（不经过sigmoid或softmax）
        if self.refine_times > 0:
            return (objectness_logits, 
                    topn_box, topn_cls,
                    topn_box_refine_list, topn_cls_refine_list,
                    topn_idx)
        else:
            return (objectness_logits,
                    topn_box, topn_cls,
                    topn_idx)
          
    def _get_objectness_single(self, gt_bboxes, output_strides, objectness_shape_list):
        """
            Args:
                gt_bboxes (Tensor): shape = (num_gt, 4), 像素值坐标（变换之后的）
                output_strides (list): [8, 16, 32, 64, 128]
                objectness_shape (list): [(h1 ,w1), ..., (hn, wn)]
            
            Returns:
                objectness_list (list): [objectness_level1, objectness_level2, ..., objectness_level5],
                    !objectness_level1 shape = (1, h_level1, w_level1)
        """
        device = gt_bboxes.device
        if len(gt_bboxes) > 0:
            objectness_list = []
            for s, (h, w) in zip(output_strides, objectness_shape_list):
                xy = torch.stack([
                    (torch.arange(h, dtype=torch.float, device=device) * s + int(s / 2))[:, None].expand(-1, w),
                    (torch.arange(w, dtype=torch.float, device=device) * s + int(s / 2))[None, :].expand(h, -1)
                    ], dim=-1)
                l = xy[..., 1] - gt_bboxes[:, None, None, 0]
                r = gt_bboxes[:, None, None, 2] - xy[..., 1]
                t = xy[..., 0] - gt_bboxes[:, None, None, 1]
                b = gt_bboxes[:, None, None, 3] - xy[..., 0]
                in_box = (l > 0) & (r > 0) & (t > 0) & (b > 0)
                objectness = torch.zeros((gt_bboxes.shape[0], h, w), dtype=torch.float, device=device)
                objectness[in_box] = torch.sqrt(torch.min(l,r)  * torch.min(t,b) / (torch.max(l,r) * torch.max(t,b)))[in_box]
                objectness, _ = torch.max(objectness, dim=0)
                objectness_list.append(objectness[None, :])
        else:
            objectness_list = [torch.zeros((1, h, w), dtype=torch.float, device=device) for (h, w) in objectness_shape_list]

        return objectness_list

    def _get_objectness(self, gt_bboxes_list, output_strides, objectness_shape_list):
        """
            Args:
                gt_bboxes_list (list): [img1_gt_bboxes_tensor, img2_gt_bboxes_tensor, ...] 
                output_strides (list): [8, 16, 32, 64, 128]
                objectness_shape_list (list): [(level1_h, level1_w), (level2_h, level2_w), ...]
            Returns:
                objectness_list (list): [level1_obj, level2_obj, ...]
                    !level1_obj shape = (num_imgs, 1, h_level1, w_level1)
        """
        
        batch_size = len(gt_bboxes_list)
        objectness_list = multi_apply(self._get_objectness_single,
                                      gt_bboxes_list,
                                      output_strides=output_strides,
                                      objectness_shape_list=objectness_shape_list)
        objectness_list = [torch.stack(objectness, dim=0) for objectness in objectness_list]
        return objectness_list

    def _get_targets_single(self, 
                            cat_bbox_pred,
                            cat_cls_pred,
                            valid_flags, 
                            gt_bboxes, 
                            gt_labels,
                            img_meta,
                            gt_bboxes_ignore=None, 
                            unmap_outputs=True):
        """
            Args:
                cat_bbox_pred (tensor): shape = (num_level * topn, 4) 
                    !4: (cx, cy, w, h) in range(0, 1)
                cat_cls_pred (tensor): shape = (num_level * topn, num_cls)
                valid_flags (tensor): shape = (num_level * topn, )
                gt_bboxes (tensor): shape = (num_gt, 4)
                gt_labels (tensor): shape = (num_gt, num_cls)
                img_meta (dict): ...
                gt_bboxes_ignore (tensor): HungarianAssigner要求该参数必须是None
                unmap_outputs (bool): Whether to map outputs back to the original
                    set of anchors.
           
            Returns:
                labels (tensor): shape = (num_level * topn, )
                labels_weights (tensor): shape = (num_level * topn, )
                bbox_gt (tensor): shape = (num_level * topn, 4)
                pos_bbox_pred (tensor): shape = (num_level * topn, 4)
                bbox_pred_weights (tensor): shape = (num_level * topn, 4)
                pos_inds (tensor): shape = (num_level * topn, )
                neg_inds (tensor): shape = (num_level * topn, )
        """
        inside_flags = torch.squeeze(valid_flags)
        if not inside_flags.any():
            return (None, ) * 7
        
        bbox_pred = cat_bbox_pred[inside_flags, :]
        #! 如果assigner为HungarianAssigner，则需要类别信息
        if self.train_cfg["assigner"]["type"] == "HungarianAssigner":
            cls_pred = cat_cls_pred[inside_flags, :]

        pos_weight = self.train_cfg["pos_weight"]

        if self.train_cfg["assigner"]["type"] == "HungarianAssigner":
            #! 将forward中bbox_pred理解为 (cx,cy,w,h), 取值为[0,1],但是是不是在[0,1]不能保证
            #! bbox_pred (cx, cy, w, h) [0,1] normalized
            #! gt_boxxes (x1, y1, x2, y2) unormalized

            # xmin = bbox_pred[..., 0:1] - bbox_pred[..., 2:3] * 0.5 
            # ymin = bbox_pred[..., 1:2] - bbox_pred[..., 3:4] * 0.5 
            # xmax = bbox_pred[..., 0:1] + bbox_pred[..., 2:3] * 0.5 
            # ymax = bbox_pred[..., 1:2] + bbox_pred[..., 3:4] * 0.5 

            # invalid_xmin = xmin < 0.
            # invalid_ymin = ymin < 0.
            # invalid_xmax = xmax > 1.0
            # invalid_ymax = ymax > 1.0
            
            # invalid_all_num = torch.logical_and(torch.logical_and(invalid_xmin, invalid_ymin),
            #                                 torch.logical_and(invalid_xmax, invalid_ymax)).sum()

            # invalid_any_num = torch.logical_or(torch.logical_or(invalid_xmin, invalid_ymin),
            #                                 torch.logical_or(invalid_xmax, invalid_ymax)).sum()

            # s = bbox_pred.shape[0]
            # print("invalid_all_num %d/%d" % (int(invalid_all_num), int(s)))
            # print("invalid_any_num %d/%d" % (int(invalid_any_num), int(s)))

            assign_result = self.assigner.assign(bbox_pred, 
                                                cls_pred, 
                                                gt_bboxes, 
                                                gt_labels, 
                                                img_meta, 
                                                gt_bboxes_ignore=gt_bboxes_ignore)
        elif self.train_cfg["assigner"]["type"] == "MaxIoUAssigner":
            #! 需要坐标是(x1, y1, x2, y2)格式的
            #! 将forward中bbox_pred(cx,cy,w,h)转化为(x1,y1,x2,y2) 
            #! bbox_pred (x1, y1, x2, y2) 需要归一化
            half_wh_bbox_pred = bbox_pred[..., 2:4] / 2
            bbox_pred = torch.cat([
                bbox_pred[..., :2] - half_wh_bbox_pred,
                bbox_pred[..., :2] + half_wh_bbox_pred
            ], dim=-1)

            img_h, img_w, _ = img_meta["img_shape"]
            factor = torch.tensor([img_w, img_h, img_w, img_h]).reshape((1,4)).cuda()
            gt_bboxes /= factor
            assign_result = self.assigner.assign(bbox_pred, 
                                                gt_bboxes, 
                                                gt_bboxes_ignore=gt_bboxes_ignore,
                                                gt_labels=gt_labels)
        else:
            raise self.train_cfg["assigner"]["type"] + "is not in (HungarianAssigner, MaxIoUAssigner)"

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        
        num_valid_bbox_pred = bbox_pred.shape[0]
        bbox_gt = bbox_pred.new_zeros([num_valid_bbox_pred, 4])
        pos_bbox_pred = torch.zeros_like(bbox_pred)
        bbox_pred_weights = bbox_pred.new_zeros([num_valid_bbox_pred, 4])
        # ! 分类用的sigmoid就是输出通道数就是类别数，如果是softmax就是类别数加1（背景，最后一类）
        labels = bbox_pred.new_full((num_valid_bbox_pred, ), self.num_classes, dtype=torch.long) 
        label_weights = bbox_pred.new_zeros(num_valid_bbox_pred, dtype=torch.float)

        pos_inds = sampling_result.pos_inds 
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes
            pos_bbox_pred[pos_inds, :] = bbox_pred[pos_inds, :]
            bbox_pred_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of bbox_pred
        if unmap_outputs:
            num_total_bbox_pred = cat_bbox_pred.size(0)
            labels = unmap(labels, num_total_bbox_pred, inside_flags)
            label_weights = unmap(label_weights, num_total_bbox_pred, inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_bbox_pred, inside_flags)
            pos_bbox_pred = unmap(pos_bbox_pred, num_total_bbox_pred, inside_flags)
            bbox_pred_weights = unmap(bbox_pred_weights, num_total_bbox_pred, inside_flags)
        
        if self.train_cfg["assigner"] == "HungarianAssigner":
            #! 最终计算损失的时候需要bbox_pred是(x1,y1,x2,y2),
            #! 而在上面的HungarianAssigner中bbox_pred还是(cx,cy,w,h),并且gt还需要归一化
            #! (cx, cy, w, h) -> (x1,y1,x2,y2)
            half_wh_bbox_pred = bbox_pred[..., 2:4] * 0.5
            bbox_pred = torch.cat([
                bbox_pred[..., :2] - half_wh_bbox_pred,
                bbox_pred[..., :2] + half_wh_bbox_pred
            ], dim=-1)
            img_h, img_w, _ = img_meta["img_shape"]
            factor = torch.tensor([img_w, img_h, img_w, img_h]).reshape((1,4)).cuda()
            bbox_gt /= factor

        return (labels, label_weights, 
                bbox_gt, 
                pos_bbox_pred, bbox_pred_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    objectness_pred_list,
                    bbox_pred_list,
                    cls_pred_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """
            计算出粗分类的标签，最后检测的分类和回归
            粗分类标签需要根据centerness来计算
            分类和回归需要先进行匹配，生成和预测框一样大小的target，在对应位置填上对应值

            Args:
                objectness_pred_list (list[tensor]): [level1_obj_tensor, level2_obj_tensor, ...]
                    level1_obj_tensor: [num_imgs, 1, H, W]
                bbox_pred_list (list[list]): [img1_bbox_list, img2_bbox_list, ...], 
                    img1_bbox_list = [level1_bbox_tensor, level2_bbox_tensor, ...], 
                    level1_bbox_tensor shape = [topn, 4]
                cls_pred_list (list[list]): [img1_cls_list, img2_cls_list, ...]
                    img1_cls_list = [level1_cls_tensor, level2_cls_tensor, ...]
                    level1_cls_tensor shape = [topn, num_cls]
                valid_flag_list (list[list]): [img1_flag_list, img2_flag_list, ...]
                    img1_flag_list = [level1_flag_tensor, level2_flag_tensor]
                    level1_flag_tensor shape = [topn, 1]
                gt_bboxes_list (list[Tensor]): [img1_box_tensor, img2_box_tensor, ...]
                    img1_box_tensor shape = [img1_num_box, 4]
                img_metas (list[dict]): [img1_meta, img2_meta, ...]
                gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                    ignored.
                gt_labels_list (list[Tensor]): [img1_label_tensor, img2_label_tensor]
                    img_label_tensor shape = [img1_num_box,]
                unmap_outputs (bool): Whether to map outputs back to the original
                    set of anchors.

            Returns:
                gt_objectness_list (list[tensor]): [level1_obj_tensor, level2_obj_tensor, ...]
                labels_list (list[tensor]): [level1_label_tensor, level2_label_tensor, ...]
                label_weights_list (list[tensor]): [level1_label_weight_tensor, level2_label_weight_tensor, ...] 
                bbox_gt_list (list[tensor]): [level1_bbox_gt_tensor, level2_bbox_gt_tensor, ...]
                bbox_pred_list (list[tensor]): [level1, level2, ...]
                bbox_pred_weights_list (list[tensor]): [level1, level2, ...]
                num_total_pos (int):
                num_total_neg (int): 
        """
        num_levels = len(self.output_strides)
        num_imgs = len(img_metas)
        
        # - objectness
        objectness_shape_list = [obj.shape[-2:] for obj in objectness_pred_list]
        gt_objectness_list = self._get_objectness(gt_bboxes_list, self.output_strides, objectness_shape_list)
        # - cls_target, reg_target
        # - concat all level bbox_pred and flag of a image to a single tensor
        for i in range(num_imgs):
            assert len(bbox_pred_list[i]) == len(valid_flag_list[i])
            bbox_pred_list[i] = torch.cat(bbox_pred_list[i], dim=0)
            cls_pred_list[i] = torch.cat(cls_pred_list[i], dim=0)
            valid_flag_list[i] = torch.cat(valid_flag_list[i], dim=0)
 
        # - compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        
        (all_labels, all_label_weights,
         all_bbox_gt, 
         all_bbox_pred, all_bbox_pred_weights,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                                     bbox_pred_list, cls_pred_list, valid_flag_list,
                                                     gt_bboxes_list, gt_labels_list,
                                                     img_metas,
                                                     gt_bboxes_ignore_list,
                                                     unmap_outputs=unmap_outputs)

        if any([labels is None for labels in all_labels]):
            return None
        
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        num_level_list = [self.topn for _ in range(num_levels)]
        labels_list = images_to_levels(all_labels, num_level_list)
        labels_weights_list = images_to_levels(all_label_weights, num_level_list)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_level_list)
        bbox_pred_list = images_to_levels(all_bbox_pred, num_level_list)
        bbox_pred_weights_list = images_to_levels(all_bbox_pred_weights, num_level_list)

        return (gt_objectness_list,
                labels_list, labels_weights_list,
                bbox_gt_list,
                bbox_pred_list, bbox_pred_weights_list,
                num_total_pos, num_total_neg)

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def get_valid_flag(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def loss(self,
             objectness_pred,
             bbox_pred,
             cls_pred,
             box_refine_list,
             cls_refine_list,
             topn_idx,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """
            Args:
                objectness_pred (list[tensor]): [level1_obj_tensor, level2_obj_tensor, ...]
                    level1_obj_tensor: [num_imgs, 1, H, W]
                bbox_pred (list[tensor]): [level1_bbox_tensor, level2_bbox_tensor, ...]
                    level1_bbox_tensor: [num_imgs, topn, 4]
                    坐标是(cx, cy, x, y), 且数值是[0,1]
                cls_pred (list[tensor]): [level1_cls_tensor, level2_cls_tensor, ...]
                    level1_cls_tensor: [num_imgs, topn, num_cls]
                box_refine_list (list[list[tensor]]): [
                    [box_refine1_scale1, box_refine2_scale1, ..., box_refinen_scale1],
                    [box_refine1_scale2, box_refine2_scale2, ..., box_refinen_scale2],
                    [], ...]
                    box_refine1_scale1: shape = [B, topn, 4]
                    ! if refine_times < 0, then box_refine_list = [None, None, None, None, None]
                cls_refine_list: [
                    [cls_refine1_scale1, cls_refine2_scale1, ..., cls_refinen_scale1],
                    [cls_refine1_scale2, cls_refine2_scale2, ..., cls_refinen_scale2],
                    [], ...]
                    cls_refine1_scale1: shape = [B, topn, num_classes]
                    ! if refine_times < 0, then cls_refine_list = [None, None, None, None, None]
                topn_idx (list[tensor]): [level1_idx_tensor, level2_idx_tensor, ...]
                    level1_idx_tensor: [num_imgs, topn, 1]
                gt_bboxes (list[tensor]): [img1_gt_bbox_tensor, img2_gt_bbox_tensor, ...]
                    坐标是(x1, y1, x2, y2)，图像中的坐标，不是feature map中的坐标
                gt_labels (list[tensor]): [img1_gt_label_tensor, img2_gt_label_tensor, ...]
                img_metas (list[dict]): [img1_meta, img2_meta, ...]
                gt_bboxes_ignore ([type], optional): [description]. Defaults to None.

            Returns:
                loss_dcit_all: {objectness_loss, bbox_loss, cls_loss}
        """
        num_imgs = len(img_metas)
        num_levels = len(objectness_pred)
        device = gt_bboxes[0].device

        bbox_pred_list = [] 
        cls_pred_list = [] 
        valid_flag_list = [] 
        for i_img, img_meta in enumerate(img_metas):
            tmp_bbox = []
            tmp_cls = []
            tmp_flag = []
            for i_lvl in range(num_levels):
                tmp_bbox.append(bbox_pred[i_lvl][i_img])
                tmp_cls.append(cls_pred[i_lvl][i_img])
        
                s = self.output_strides[i_lvl]
                feat_h, feat_w = objectness_pred[i_lvl].shape[-2:]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int((h + s - 1) / s), feat_h)
                valid_feat_w = min(int((w + s - 1) / s), feat_w)
                # - flags shape = [feat_h * feat_w, ]
                flags = self.get_valid_flag((feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                # !  topn_idx[i_lvl][i_img] shape = (topn, 1)
                flags = torch.gather(flags.view(-1, 1), 0, topn_idx[i_lvl][i_img])
                # - flags shape = [topn, 1]
                tmp_flag.append(flags)
            bbox_pred_list.append(tmp_bbox)
            cls_pred_list.append(tmp_cls)
            valid_flag_list.append(tmp_flag)

        (gt_objectness_list,
         labels_list, label_weights_list,
         bbox_gt_list, bbox_pred_list, bbox_pred_weights_list,
         num_total_pos, num_total_neg) = self.get_targets(objectness_pred,
                                                          bbox_pred_list,
                                                          cls_pred_list,
                                                          valid_flag_list,
                                                          gt_bboxes,
                                                          img_metas,
                                                          gt_bboxes_ignore,
                                                          gt_labels)
        num_level_list = [self.topn for _ in range(num_levels)]                                                  
        cls_pred_list = images_to_levels(cls_pred_list, num_level_list)

        num_total_samples = (num_total_pos + num_total_neg) if self.sampling else num_total_pos
        loss_obj, loss_bbox, loss_cls = multi_apply(self.loss_single,
                                                    objectness_pred,
                                                    bbox_pred_list,
                                                    cls_pred_list,
                                                    box_refine_list,
                                                    cls_refine_list,
                                                    gt_objectness_list,
                                                    bbox_gt_list, bbox_pred_weights_list,
                                                    labels_list, label_weights_list,
                                                    num_total_samples=num_total_samples)

        loss_dict_all = {
            'loss_obj': loss_obj,
            'loss_bbox': loss_bbox,
            'loss_cls': loss_cls
        }

        return loss_dict_all

    def loss_single(self, 
                    objectness_pred,
                    bbox_pred,
                    cls_pred,
                    bbox_refine,
                    cls_refine,
                    objecness_gt,
                    bbox_gt, bbox_weights,
                    cls_gt, cls_weights,
                    num_total_samples):
        """
            在不同scale下进行

            Args:
                objectness_pred (tensor): shape = [num_imgs, 1, H, W]
                bbox_pred (tensor): shape = [num_imgs, topn, 4]
                cls_pred (tensor): shape = [num_imgs, topn, num_cls]
                bbox_refine (list[tensor]): [bbox_refine1, bbox_refine2, ...]
                    bbox_refine1 shape = [B, topn, 4]
                cls_refine(list[tensor]): [cls_refine1, cls_refine2, ...]
                    cls_refine1 shape = [B, topn, num_classes]
                objecness_gt (tensor): shape = [num_imgs, 1, H, W]
                bbox_gt (tensor): shape = [num_imgs, topn, 4]
                bbox_weights (tensor): [num_imgs, topn, 4]
                cls_gt (tensor): shape = [num_imgs, topn]
                cls_weights (tensor): shape = [num_imgs, topn]
                num_total_samples (int): 

            Returns:
                loss_obj, loss_bbox, loss_cls
        """
        # objectness_pred = torch.reshape(objectness_pred, (-1, 1))
        # objecness_gt = torch.reshape(objecness_gt, (-1, 1))
        loss_obj = self.loss_obj(objectness_pred, objecness_gt)
        
        bbox_pred = torch.reshape(bbox_pred, (-1, 4))
        bbox_gt = torch.reshape(bbox_gt, (-1, 4))
        bbox_weights = torch.reshape(bbox_weights, (-1, 4))
        loss_bbox = self.loss_bbox(bbox_pred,
                                   bbox_gt,
                                   bbox_weights,
                                   avg_factor=num_total_samples)
    
        cls_pred = cls_pred.reshape((-1, self.cls_out_channels))
        cls_gt = cls_gt.reshape((-1,))
        cls_weights = cls_weights.reshape((-1,))
        loss_cls = self.loss_cls(cls_pred,
                                 cls_gt,
                                 cls_weights,
                                 avg_factor=num_total_samples)

        if self.refine_times > 0 and bbox_refine[0] != None and cls_refine[0] != None:
            for i, h in enumerate(self.refineHead_list):
                tmp_bbox_refine = torch.reshape(bbox_refine[i], (-1, 4))
                tmp_cls_refine = torch.reshape(cls_refine[i], (-1, self.cls_out_channels))

                loss_bbox_refine, loss_cls_refine = h.loss_single(tmp_bbox_refine, 
                                                                  tmp_cls_refine,
                                                                  bbox_gt, bbox_weights,
                                                                  cls_gt, cls_weights,
                                                                  num_total_samples) 
                loss_bbox += loss_bbox_refine
                loss_cls += loss_cls_refine

        return loss_obj, loss_bbox, loss_cls

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if self.refine_times == 0:
            (objectness, topn_box, topn_cls, topn_idx) = outs
            topn_box_refine_list = [None for _ in range(len(self.output_strides))]
            topn_cls_refine_list = [None for _ in range(len(self.output_strides))]
            outs = (objectness, topn_box, topn_cls, topn_box_refine_list, topn_cls_refine_list, topn_idx)

        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)

        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def get_bboxes(self, bbox_pred, cls_pred, img_metas, cfg=None, rescale=False, with_nms=False):
        """

            Args:
                bbox_pred (list[tensor]): [level1_bbox_pred, level2_bbox_pred, ...], 
                    level1_bbox_pred shape = [num_imgs, topn, 4]
                    !4: [cx, cy, w, h] normalized
                cls_pred (list[tensor]): [level1_cls_pred, level2_cls_pred, ...]
                    level1_bbox_pred shape = [num_imgs, topn, num_cls]
                img_metas (list[dict]): [img1_meta, img2_meta, ...]
                cfg ([type], optional): [description]. Defaults to None.
                rescale (bool, optional): [description]. Defaults to False.
                with_nms (bool, optional): [description]. Defaults to False.

            Returns:
                result_list: 
        """
        device = bbox_pred[0].device
        num_imgs = len(img_metas)

        # - bbox_pred shape = [B, num_level * topn, 4]
        bbox_pred = torch.cat(bbox_pred, dim=1)
        # - cls_pred shape = [B, num_level * topn , num_cls]
        cls_pred = torch.cat(cls_pred, dim=1)
        # - [cx, cy, w, h] -> [xmin, ymin, xmax, ymax]
        half_wh = bbox_pred[..., 2:] / 2
        bbox_pred = torch.cat([
            bbox_pred[..., :2] - half_wh,
            bbox_pred[..., :2] + half_wh
        ], dim=-1)

        if self.use_sigmoid_cls:
            cls_pred = cls_pred.sigmoid()
        else:
            cls_pred = cls_pred.softmax(dim=-1)
 
        result_list = []
        for i in range(num_imgs):
            result_list.append(
                self._get_bboxes_single(bbox_pred[i],
                                        cls_pred[i],
                                        img_metas[i],
                                        cfg, rescale, with_nms))
        return result_list

    def _get_bboxes_single(self, bbox_pred, cls_pred, img_meta, cfg, rescale=False, with_nms=False):
        """

            Args:
                bbox_pred (tensor): shape = [num_level * topn, 4] 
                    4: [x_min, ymin, xmax, ymax]
                    !normalized
                cls_pred (tensor): shape = [num_level * topn, num_cls] 
                img_meta (dict): 
                cfg ([type]): [description]
                rescale (bool, optional): [description]. Defaults to False.
                with_nms (bool, optional): [description]. Defaults to False.
            
            Returns: 
                det_bboxes, det_labels
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(bbox_pred) == len(cls_pred)
        img_h, img_w = img_meta["img_shape"]
        # - unormalized
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).view((1, 4)).cuda()
        bbox_pred *= scale

        x1 = bbox_pred[:, 0].clamp(min=0, max=img_w)
        y1 = bbox_pred[:, 1].clamp(min=0, max=img_h)
        x2 = bbox_pred[:, 2].clamp(min=0, max=img_w)
        y2 = bbox_pred[:, 3].clamp(min=0, max=img_h)
        bbox_pred = torch.stack([x1, y1, x2, y2], dim=-1)
        
        scale_factor = img_meta["scale_factor"]
        if rescale:
            bbox_pred /= bbox_pred.new_tensor(scale_factor)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = cls_pred.new_zeros(cls_pred.shape[0], 1)
            cls_pred = torch.cat([cls_pred, padding], dim=-1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(bbox_pred, cls_pred,
                                                 cfg.score_thr, cfg.nms,
                                                 cfg.max_per_img)  
            return det_bboxes, det_labels
        else:
            cls_score, cls_pred_idx = cls_pred.max(axis=-1)
            bbox_pred = torch.cat([bbox_pred, cls_score.unsqueeze(-1)], dim=-1)
            return bbox_pred, cls_pred_idx
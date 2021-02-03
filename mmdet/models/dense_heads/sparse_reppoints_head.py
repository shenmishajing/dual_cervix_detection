import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import (build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class SparseRepPointsHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 num_points=9,
                 top_k=10,
                 stacked_linears=3,
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
                 **kwargs):

        self.stacked_linears = stacked_linears
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.top_k = top_k

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
        
        self.cfg_loss_obj = loss_obj

    def _init_layers(self):
        """Initialize layers of the head."""

        self.relu = nn.ReLU(inplace=True).cuda()

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
                    norm_cfg=self.norm_cfg).cuda())
        
        pts_out_dim = 2 * self.num_points
        self.offset_conv = nn.Conv2d(self.feat_channels,
                                      self.point_feat_channels,
                                      3, 1, 1).cuda()
        self.offset_out = nn.Conv2d(self.point_feat_channels,
                                     pts_out_dim,
                                     1, 1, 0).cuda()

        self.objectness_conv = nn.Conv2d(self.feat_channels,
                                         self.feat_channels,
                                         3, 1, 1).cuda()
        self.objectness_out = nn.Conv2d(self.point_feat_channels,
                                        1,
                                        1, 1, 0).cuda()
        self.objectness_sigmoid_out = nn.Sigmoid().cuda()

        self.encode_points_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = pts_out_dim if i == 0 else self.point_feat_channels
            self.encode_points_convs.append(
                nn.Conv1d(chn,
                          self.point_feat_channels,
                          3, 1, 1).cuda())

        concat_feat_channels = (self.num_points + 1) * self.point_feat_channels
        self.concat_feat_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            self.concat_feat_convs.append(
                nn.Conv1d(concat_feat_channels,
                          concat_feat_channels,
                          3, 1, 1).cuda())

        self.reg_conv = nn.Conv1d(concat_feat_channels,
                                  concat_feat_channels // 2,
                                  3, 1, 1).cuda()
        self.reg_out = nn.Conv1d(concat_feat_channels // 2,
                                 4,
                                 1, 1, 0).cuda()
        self.reg_sigmoid_out = nn.Sigmoid().cuda()

        self.cls_conv = nn.Conv1d(concat_feat_channels,
                                  concat_feat_channels // 2,
                                  3, 1, 1).cuda()
        self.cls_out = nn.Conv1d(concat_feat_channels // 2,
                                 self.cls_out_channels,
                                 1, 1, 0).cuda()

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
            normal_init(m, std=0.01)
        
        normal_init(self.reg_conv, std=0.01)
        normal_init(self.reg_out, std=0.01)

        normal_init(self.cls_conv, std=0.01)
        normal_init(self.cls_out, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        # - Forward feature map of a single FPN level.
        # - dcn_base_offset.shape = [1, 18, 1, 1] 18: (-1, -1), (-1, 0)...
        h, w = x.shape[-2: ]
        dcn_base_offset = self.dcn_base_offset.type_as(x).unsqueeze(dim=2)
        dcn_base_offset = dcn_base_offset.reshape((-1, 2))
        dcn_base_offset = torch.cat([
            dcn_base_offset[:, 0:1] / h,
            dcn_base_offset[:, 1:2] / w
        ], dim=-1)
        dcn_base_offset = dcn_base_offset.reshape((1, -1, 1, 1))

        feat = x 
        for conv in self.convs:
            feat = conv(feat)
        
        offset = self.offset_out(
            self.relu(self.offset_conv(feat)))
        #! dcn_base_offset，如果3*3的网格，中心坐标是(0,0)，中心坐标加上dcn_base_offset可以得到网格其他位置坐标
        #! 取值是[[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
        offset = offset + dcn_base_offset

        if self.cfg_loss_obj["type"] == "CrossEntropyLoss":
            objectness_logits = self.objectness_out(self.objectness_conv(feat))
            objectness = self.objectness_sigmoid_out(objectness_logits)
        else:
            objectness = self.objectness_sigmoid_out(
                self.objectness_out(self.objectness_conv(feat)))

        # - topn_objectness shape = [B, 1, topn], top_idx shape = [B, 1, topn]
        topn_objectness, topn_idx = torch.topk(objectness.view((objectness.shape[0], 1, -1)), self.top_k, dim=-1)
        h, w = objectness.shape[-2:]
        # - topn_grid_coord shape = [B, 2, topn] 
        topn_grid_coord = torch.stack([topn_idx[..., -2] / w, topn_idx[..., -1] % w], axis=1)
        # - topn_offset shape = [B, 2 * num_points, topn]
        topn_offset = torch.gather(offset.view((offset.shape[0], 2 * self.num_points, -1)), -1, topn_idx)
        #! 将grid坐标由[0, feat_h/feat_w] 变成 [-1, 1]
        topn_grid_coord = torch.cat([
            topn_grid_coord[:, 0:1] / h,
            topn_grid_coord[:, 1:2] / w
        ], dim=1)
        topn_grid_coord =  2.0 * topn_grid_coord - 1.0
        topn_points = topn_offset + topn_grid_coord.repeat((1, self.num_points, 1))
         
        topn_points_transposed = topn_points.view((-1, self.num_points, 2, self.top_k)).transpose(-1, -2)
        #! 要求给出索引范围必须（需要标准化）是[-1，1]，
        # print("topn_points", topn_points_transposed.min(), topn_points_transposed.max(), topn_points_transposed.mean(), topn_points_transposed.std())
        topn_feat = torch.nn.functional.grid_sample(feat, topn_points_transposed)
        topn_feat = topn_feat.view((topn_feat.shape[0] , self.num_points * self.point_feat_channels, -1))

        topn_points_encoded = topn_points
        for conv in self.encode_points_convs:
            topn_points_encoded = conv(topn_points_encoded)

        topn_feat_concat = torch.cat([topn_feat, topn_points_encoded], axis=-2)
        for conv in self.concat_feat_convs:
            topn_feat_concat = conv(topn_feat_concat)
        
        topn_box = self.reg_out(self.reg_conv(topn_feat_concat))
        # topn_box = self.reg_sigmoid_out(topn_box)
        topn_cls = self.cls_out(self.cls_conv(topn_feat_concat))
        
        # - [B, 1, H, W] -> [B, H, W]
        # objectness_logits = torch.squeeze(objectness_logits)
        # - [B, 4, topn] -> [B, topn ,4]
        topn_box = torch.transpose(topn_box, -2, -1)
        topn_cls = torch.transpose(topn_cls, -2, -1)
        topn_idx = torch.transpose(topn_idx, -2, -1)

        if self.cfg_loss_obj["type"] == "CrossEntropyLoss":
            return objectness_logits, topn_box, topn_cls, topn_idx
        else:
            return objectness, topn_box, topn_cls, topn_idx

    def _get_objectness_single(self, gt_bboxes, img_meta, output_strides, objectness_shape_list):
        """
            Args:
                gt_bboxes (Tensor): shape = (num_gt, 4), 像素值坐标（变换之后的）
                img_meta (dict):
                    {   
                        'filename': 'data/coco/train2017/000000251577.jpg',
                        'ori_filename': '000000251577.jpg',
                        'ori_shape': (289, 500, 3),
                        'img_shape': (770, 1333, 3),
                        'pad_shape': (800, 1344, 3),
                        'scale_factor': array([2.666    , 2.6643598, 2.666    , 2.6643598], dtype=float32),
                        'flip': False,
                        'flip_direction': None, 
                        'img_norm_cfg': 
                            {
                                'mean': array([123.675, 116.28 , 103.53 ], dtype=float32),
                                'std': array([58.395, 57.12 , 57.375], dtype=float32),
                                'to_rgb': True
                            },
                        'batch_input_shape': (800, 1344)
                    }
                output_strides (list): [8, 16, 32, 64, 128]
                objectness_shape (list): [(h1 ,w1), ..., (hn, wn)]
            
            Returns:
                objectness_list (list): [objectness_level1, objectness_level2, ..., objectness_level5],
                    !objectness_level1 shape = (1, h_level1, w_level1)
        """
        device = gt_bboxes.device
        img_h, img_w = img_meta["batch_input_shape"]
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
            objectness_list = [torch.zeros((1, h, w), dtype=torch.float, device=device) for (h,w) in objectness_shape_list]
        
        # print("objectness_single", [obj.shape for obj in objectness_list])
        return objectness_list

    def _get_objectness(self, gt_bboxes_list, img_metas, output_strides, objectness_shape_list):
        """
            Args:
                gt_bboxes_list (list): [img1_gt_bboxes_tensor, img2_gt_bboxes_tensor, ...] 
                img_metas (list): [img1_metas, img2_metas, ...]
                output_strides (list): [8, 16, 32, 64, 128]
                objectness_shape_list (list): [(level1_h, level1_w), (level2_h, level2_w), ...]
            Returns:
                objectness_list (list): [level1_obj, level2_obj, ...]
                    !level1_obj shape = (num_imgs, 1, h_level1, w_level1)
        """
        
        batch_size = len(gt_bboxes_list)
        objectness_list =  multi_apply(self._get_objectness_single,
                                       gt_bboxes_list,
                                       img_metas,
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
                            label_channels=1,
                            unmap_outputs=True):
        """
            Args:
                cat_bbox_pred (tensor): shape = (num_level * topn, 4) 
                    !4: (cx, cy, w, h) in range(0, 1)
                cat_cls_pred (tensor): shape = (num_level * topn, num_cls)
                valid_flags (tensor): shape = (num_level * topn,)
                gt_bboxes (tensor): shape = (num_gt, 4)
                gt_labels (tensor): shape = (num_gt, label_channels)
                img_meta (dict): ...
                gt_bboxes_ignore (tensor): HungarianAssigner要求该参数必须是None
                label_channels (int): channel of label
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
        cls_pred = cat_cls_pred[inside_flags, :]

        # TODO 改掉下面的取参数的用法
        # pos_weight = self.train_cfg.pos_weight
        pos_weight = self.train_cfg["pos_weight"]

        if self.train_cfg["assigner"]["type"] == "HungarianAssigner":
            #! 将forward中bbox_pred理解为 (x1, y1, x2, y2), 并需要通过feature map 的shape进行归一化
            #! bbox_pred (cx, cy, w, h) [0,1] normalized
            #! gt_boxxes (x1, y1, x2, y2) unormalized
            h, w = img_meta["batch_input_shape"]
            factor = [[int((h + s - 1 ) / s), int((w + s - 1) / s)] * 2 for s in self.output_strides]
            topn = cat_bbox_pred.size(0) // len(self.output_strides)
            factor = torch.tensor(factor, dtype=torch.float).repeat_interleave(topn, dim=0).cuda()
            bbox_pred = bbox_pred / factor
            bbox_pred = torch.cat([
                (bbox_pred[..., 2:4] + bbox_pred[..., :2]) / 2,
                bbox_pred[..., 2:4] - bbox_pred[..., :2]
            ],dim=-1)
            assign_result = self.assigner.assign(bbox_pred, 
                                                cls_pred, 
                                                gt_bboxes, 
                                                gt_labels, 
                                                img_meta, 
                                                gt_bboxes_ignore=gt_bboxes_ignore)
        elif self.train_cfg["assigner"]["type"] == "MaxIoUAssigner":
            #! 需要坐标是(x1, y1, x2, y2)格式的
            #! 将forward中bbox_pred理解为 (x1, y1, x2, y2) 
            #! bbox_pred (x1, y1, x2, y2) 得到的最终坐标需要乘上output_strides
            topn = cat_bbox_pred.size(0) // len(self.output_strides)
            scale = torch.tensor(self.output_strides).repeat_interleave(topn, dim=0).reshape((-1, 1)).cuda()
            cat_bbox_pred = cat_bbox_pred * scale

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
        labels = bbox_pred.new_full((num_valid_bbox_pred, ), self.cls_out_channels, dtype=torch.long) 
        label_weights = bbox_pred.new_zeros(num_valid_bbox_pred, dtype=torch.float)

        pos_inds = sampling_result.pos_inds 
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
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
            #! 最终计算损失的时候需要bbox_pred是(x1,y1,x2,y2)
            #! (cx, cy, w, h) -> (x1,y1,x2,y2)
            half_wh_bbox_pred = bbox_pred[..., 2:4] / 2
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
                    label_channels=1,
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
                stage (str): `init` or `refine`. Generate target for init stage or
                    refine stage
                label_channels (int): Channel of label.
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
        
        # - objecness
        objectness_shape_list = [obj.shape[-2:] for obj in objectness_pred_list]
        gt_objectness_list = self._get_objectness(gt_bboxes_list, img_metas, self.output_strides, objectness_shape_list)
        # - cls_target, reg_target
        #- concat all level bbox_pred and flag to a single tensor
        for i in range(num_imgs):
            assert len(bbox_pred_list[i]) == len(valid_flag_list[i])
            bbox_pred_list[i] = torch.cat(bbox_pred_list[i], dim=0)
            cls_pred_list[i] = torch.cat(cls_pred_list[i], dim=0)
            valid_flag_list[i] = torch.cat(valid_flag_list[i], dim=0)
 
        # - compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None] * num_imgs
        if gt_labels_list is None:
            gt_labels_list = [None] * num_imgs
        
        (all_labels, all_label_weights,
         all_bbox_gt, 
         all_bbox_pred, all_bbox_pred_weights,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_targets_single,
                                                     bbox_pred_list, cls_pred_list, valid_flag_list,
                                                     gt_bboxes_list, gt_labels_list,
                                                     img_metas,
                                                     gt_bboxes_ignore_list,
                                                     label_channels=label_channels,
                                                     unmap_outputs=unmap_outputs)

        if any([labels is None for labels in all_labels]):
            return None
        
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        num_level_list = [self.top_k for _ in range(num_levels)]
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
                topn_idx (list[tensor]): [level1_idx_tensor, level2_idx_tensor, ...]
                    level1_idx_tensor: [num_imgs, topn, 1]
                gt_bboxes (list[tensor]): [img1_gt_bbox_tensor, img2_gt_bbox_tensor, ...]
                    坐标是(x1,y1,x2,y2)，图像中的坐标，不是feature map中的坐标
                gt_labels (list[tensor]): [img1_gt_label_tensor, img2_gt_label_tensor, ...]
                img_metas (list[dict]): [img1_meta, img2_meta, ...]
                gt_bboxes_ignore ([type], optional): [description]. Defaults to None.

            Returns:
                loss_dcit_all: {objectness_loss, bbox_loss, cls_loss}
        """
        num_imgs = len(img_metas)
        num_levels = len(objectness_pred)
        device = gt_bboxes[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

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
                # !  topn_idx[i_lvl][i_img] shape = (topn, 1)
                flags = self.get_valid_flag((feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
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
                                                          gt_labels,
                                                          label_channels)
        num_level_list = [self.top_k for _ in range(num_levels)]                                                  
        cls_pred_list = images_to_levels(cls_pred_list, num_level_list)

        num_total_samples = (num_total_pos + num_total_neg) if self.sampling else num_total_pos
        loss_obj, loss_bbox, loss_cls = multi_apply(self.loss_single,
                                                    objectness_pred,
                                                    bbox_pred_list,
                                                    cls_pred_list,
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
                objecness_gt (tensor): shape = [num_imgs, 1, H, W]
                bbox_gt (tensor): shape = [num_imgs, topn, 4]
                bbox_weights (tensor): [num_imgs, topn, 4]
                cls_gt (tensor): shape = [num_imgs, topn]
                cls_weights (tensor): shape = [num_imgs, topn]
                num_total_samples (int): [description]

            Returns:
                loss_obj, loss_bbox, loss_cls
        """
        objectness_pred = torch.reshape(objectness_pred, (-1, 1))
        objecness_gt = torch.reshape(objecness_gt, (-1, 1))
        loss_obj = self.loss_obj(objectness_pred, objecness_gt)
        
        # print("loss bbox_pred", bbox_pred.min(), bbox_pred.max())
        # print("loss bbox_gt", bbox_gt.min(), bbox_gt.max())

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
        #TODO gt box [x1,y1,x2,y2]-> [cx, cy, w, h], box_pred
        loss_cls = self.loss_cls(cls_pred,
                                 cls_gt,
                                 cls_weights,
                                 avg_factor=num_total_samples)

        return loss_obj, loss_bbox, loss_cls

    def get_bboxes(self, bbox_pred, cls_pred, img_metas, cfg=None, rescale=False, with_nms=False):
        """

        Args:
            bbox_pred (list[tensor]): [level1_bbox_pred, level2_bbox_pred, ...], 
                level1_bbox_pred shape = [num_imgs, topn, 4]
                4: [cx, cy, w, h]
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

        bbox_pred = torch.cat(bbox_pred, dim=1)
        cls_pred = torch.cat(cls_pred, dim=1)
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
        img_shape = img_meta["img_shape"]
        x1 = bbox_pred[:, 0].clamp(min=0, max=img_shape[1])
        y1 = bbox_pred[:, 1].clamp(min=0, max=img_shape[0])
        x2 = bbox_pred[:, 2].clamp(min=0, max=img_shape[1])
        y2 = bbox_pred[:, 3].clamp(min=0, max=img_shape[0])
        bbox_pred = torch.stack([
            x1,y1,x2,y2
        ],dim=-1)
        
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
            return bbox_pred, cls_pred
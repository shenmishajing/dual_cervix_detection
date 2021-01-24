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
                 point_feat_channels=256,
                 num_points=9,
                 top_k=100,
                 stacked_linears=3,
                 gradient_mul=0.1,
                 output_strides=[8, 16, 32, 64, 128],
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
        self.output_strides = output_strides

        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # - use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                #- A pseudo sampler that does not do sampling actually.Directly returns the positive and negative indices  of samples.
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(self.train_cfg.sampler)
        
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

        return objectness_list

    def _get_objectness(self, gt_bboxes_list, img_metas, output_strides):
        """
            Args:
                gt_bboxes_list (list): [img1_gt_bboxes_tensor, img2_gt_bboxes_tensor, ...] 
                img_metas (list): [img1_metas, img2_metas, ...]
                output_strides (list): [8, 16, 32, 64, 128]
            
            Returns:
                objectness_list (list): [img1_objectness_list, img2_objectness_list, ...]
                    !img1_objectness_level1 shape = (1, h1, w1)
        """
        objectness_shape_list = [
            (int((img_metas[0]["batch_input_shape"][0] + s - 1) / s), int((img_metas[0]["batch_input_shape"][1] + s -1) / s)) for s in self.output_strides
        ]
        batch_size = len(gt_bboxes_list)
        return multi_apply(self._get_objectness_single,
                           gt_bboxes_list,
                           img_metas,
                           [output_strides] * batch_size,
                           [objectness_shape_list] * batch_size)

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
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        
        bbox_pred = cat_bbox_pred[inside_flags, :]
        cls_pred = cat_cls_pred[inside_flags, :]

        pos_weight = self.train_cfg.pos_weight
        assign_result = self.assigner.assign(bbox_pred, 
                                             cls_pred, 
                                             gt_bboxes, 
                                             gt_labels, 
                                             img_meta, 
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        
        num_valid_bbox_pred = bbox_pred.shape[0]
        bbox_gt = bbox_pred.new_zeros([num_valid_bbox_pred, 4])
        pos_bbox_pred = torch.zeros_like(bbox_pred)
        bbox_pred_weights = bbox_pred.new_zeros([num_valid_bbox_pred, 4])
        labels = bbox_pred.new_full((num_valid_bbox_pred, ), self.num_classes, dtype=torch.long) 
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

        return (labels, label_weights, 
                bbox_gt, 
                pos_bbox_pred, bbox_pred_weights,
                pos_inds, neg_inds)

    def get_targets(self,
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
                bbox_pred_list (list[list]): Multi level bboxes of each image.
                cls_pred_list (list[list]): Multi level cls of each image.
                valid_flag_list (list[list]): Multi level valid flags of each
                    image.
                gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
                img_metas (list[dict]): Meta info of each image.
                gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                    ignored.
                gt_labels_list (list[Tensor]): Ground truth labels of each box.
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
        gt_objectness_list = self._get_objectness(gt_bboxes_list, img_metas, output_strides)
        gt_objectness_list = images_to_levels(gt_objectness_list, num_levels)

        # - cls_target, reg_target
        #- concat all level bbox_pred and flag to a single tensor
        for i in range(num_imgs):
            assert len(bbox_pred_list[i]) == len(valid_flag_list[i])
            bbox_pred_list[i] = torch.cat(bbox_pred_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        
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
                                                     gt_bboxes_list, img_metas,
                                                     gt_bboxes_ignore_list, gt_labels_list,
                                                     label_channels,
                                                     unmap_outputs)
        if any([labels is None for labels in all_labels]):
            return None
        
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        labels_list = images_to_levels(all_labels, num_levels)
        labels_weights_list = images_to_levels(all_label_weights, num_levels)
        bbox_gt_list = images_to_levels(all_bbox_gt, num_levels)
        bbox_pred_list = images_to_levels(all_bbox_pred, num_levels)
        bbox_pred_weights_list = images_to_levels(all_bbox_pred_weights, num_levels)

        return (gt_objectness_list,
                labels_list, label_weights_list,
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
            loss

            Args:
                objectness_pred (list[tensor]): [level1_obj_tensor, level2_obj_tensor, ...]
                bbox_pred (list[tensor]): [level1_bbox_tensor, level2_bbox_tensor, ...]
                cls_pred (list[tensor]): [level1_cls_tensor, level2_cls_tensor, ...]
                topn_idx (list[tensor]): [level1_idx_tensor, level2_idx_tensor, ...]
                gt_bboxes (list[tensor]): [img1_gt_bbox_tensor, img2_gt_bbox_tensor, ...]
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

        bbox_pred_list = [[] * num_imgs]
        cls_pred_list = [[] * num_imgs]
        valid_flag_list = [[] * num_imgs]
        for i_img, img_meta in enumerate(img_metas):
            for i_lvl in range(num_levels):
                bbox_pred_list[i_img].append(bbox_pred[i_lvl][i_img])
                cls_pred_list[i_img].append(cls_pred[i_lvl][i_img])
        
                s = self.output_strides[i_lvl]
                feat_h, feat_w = objectness_pred[i_lvl].shape[-2:]
                h,w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int((h + s - 1) / s), feat_h)
                valid_feat_w = min(int((w + s - 1) / s), feat_w)
                flags = self.get_valid_flag((feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                valid_flag_list[i_img].append(flags)
        

        (gt_objectness_list,
         labels_list, label_weights_list,
         bbox_gt_list, bbox_pred_list, bbox_pred_weights_list,
         num_total_pos, num_total_neg) = self.get_targets(bbox_pred_list,
                                                          cls_pred_list,
                                                          valid_flag_list,
                                                          gt_bboxes,
                                                          img_metas,
                                                          gt_bboxes_ignore,
                                                          gt_labels,
                                                          label_channels)
        


        return loss_dict_all

    def loss_single(self, 
                    objectness_pred,
                    bbox_pred,
                    cls_pred,
                    objecness_gt, objectness_weights,
                    bbox_gt, bbox_weights,
                    cls_gt, cls_weights):


        return loss_obj, loss_bbox, loss_cls

    def get_bboxes(self):

        pass

    def _get_bboxes_single(self):


        pass
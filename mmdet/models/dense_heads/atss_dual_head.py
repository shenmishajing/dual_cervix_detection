import math
import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead


@HEADS.register_module()
class ATSSDualHead(ATSSHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    ATSS head structure is similar with FCOS, however ATSS use anchor boxes
    and assign label by Adaptive Training Sample Selection instead max-iou.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(self,
                 loss_offset = dict(
                     type = 'SmoothL1Loss', beta = 1.0 / 9.0, loss_weight = 1.0),
                 **kwargs):
        super(ATSSDualHead, self).__init__(**kwargs)
        self.loss_offset = build_loss(loss_offset)

    def _init_layers(self):
        super(ATSSDualHead, self)._init_layers()
        self.acid_iodine_offset_conv = nn.ModuleList([
            nn.Conv2d(self.feat_channels * 2, self.feat_channels, 3, padding = 1),
            nn.Conv2d(self.feat_channels, 2, 3, padding = 1)
        ])
        self.acid_iodine_deform_conv = DeformConv2d(self.feat_channels, self.feat_channels, 3, padding = 1)
        self.iodine_acid_offset_conv = nn.ModuleList([
            nn.Conv2d(self.feat_channels * 2, self.feat_channels, 3, padding = 1),
            nn.Conv2d(self.feat_channels, 2, 3, padding = 1)
        ])
        self.iodine_acid_deform_conv = DeformConv2d(self.feat_channels, self.feat_channels, 3, padding = 1)

    def forward_train(self,
                      acid_feats, iodine_feats,
                      img_metas,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels = None, iodine_gt_labels = None,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      acid_proposals = None, iodine_proposals = None,
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
        outs = self(acid_feats, iodine_feats)
        losses = self.loss(*outs,
                           img_metas,
                           acid_gt_bboxes, iodine_gt_bboxes,
                           acid_gt_labels = acid_gt_labels, iodine_gt_labels = iodine_gt_labels,
                           acid_gt_bboxes_ignore = acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore = iodine_gt_bboxes_ignore, )
        if acid_proposals is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, acid_proposals, iodine_proposals)
            return losses, proposal_list

    def simple_test(self, acid_feats, iodine_feats, img_metas, rescale = False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        return self.simple_test_bboxes(acid_feats, iodine_feats, img_metas, rescale = rescale)

    def simple_test_bboxes(self, acid_feats, iodine_feats, img_metas, rescale = False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(acid_feats, iodine_feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale = rescale)
        return results_list

    def forward(self, acid_feats, iodine_feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * 4.
        """
        return multi_apply(self.forward_single, acid_feats, iodine_feats, self.scales)

    def _bbox_forward(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
                centerness (Tensor): Centerness for a single scale level, the
                    channel number is (N, num_anchors * 1, H, W).
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def forward_single(self, acid_feats, iodine_feats, scale):
        acid_iodine_feats = torch.cat([acid_feats, iodine_feats], dim = 1)
        acid_iodine_offset = acid_iodine_feats
        for m in self.acid_iodine_offset_conv:
            acid_iodine_offset = m(acid_iodine_offset)
        acid_feats = acid_feats + self.acid_iodine_deform_conv(iodine_feats, acid_iodine_offset.repeat(1, 9, 1, 1))
        acid_cls_score, acid_bbox_pred, acid_centerness = self._bbox_forward(acid_feats, scale)

        iodine_acid_feats = torch.cat([iodine_feats, acid_feats], dim = 1)
        iodine_acid_offset = iodine_acid_feats
        for m in self.iodine_acid_offset_conv:
            iodine_acid_offset = m(iodine_acid_offset)
        iodine_feats = iodine_feats + self.iodine_acid_deform_conv(acid_feats, iodine_acid_offset.repeat(1, 9, 1, 1))
        iodine_cls_score, iodine_bbox_pred, iodine_centerness = self._bbox_forward(iodine_feats, scale)

        return (acid_cls_score, iodine_cls_score,
                acid_bbox_pred, iodine_bbox_pred,
                acid_centerness, iodine_centerness,
                acid_iodine_offset, iodine_acid_offset)

    def loss_single(self, anchors,
                    acid_cls_score, iodine_cls_score,
                    acid_bbox_pred, iodine_bbox_pred,
                    acid_centerness, iodine_centerness,
                    acid_offset, iodine_offset,
                    acid_labels, acid_label_weights, acid_bbox_targets, acid_offsets,
                    iodine_labels, iodine_label_weights, iodine_bbox_targets, iodine_offsets,
                    num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            num_total_samples (int): Number os positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        anchors = anchors.reshape(-1, 4)

        # acid
        acid_cls_score = acid_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        acid_bbox_pred = acid_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        acid_centerness = acid_centerness.permute(0, 2, 3, 1).reshape(-1)
        acid_offset = acid_offset.permute(0, 2, 3, 1).reshape(-1, 2)
        acid_bbox_targets = acid_bbox_targets.reshape(-1, 4)
        acid_labels = acid_labels.reshape(-1)
        acid_label_weights = acid_label_weights.reshape(-1)
        acid_offsets = acid_offsets.reshape(-1, 2)

        # classification loss
        acid_loss_cls = self.loss_cls(
            acid_cls_score, acid_labels, acid_label_weights, avg_factor = num_total_samples)

        # offset loss
        stride = abs(anchors[1, 0] - anchors[0, 0])
        acid_loss_offset = self.loss_offset(acid_offset, acid_offsets / stride)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        acid_pos_inds = ((acid_labels >= 0)
                         & (acid_labels < bg_class_ind)).nonzero().squeeze(1)

        if len(acid_pos_inds) > 0:
            acid_pos_bbox_targets = acid_bbox_targets[acid_pos_inds]
            acid_pos_bbox_pred = acid_bbox_pred[acid_pos_inds]
            acid_pos_anchors = anchors[acid_pos_inds]
            acid_pos_centerness = acid_centerness[acid_pos_inds]

            acid_centerness_targets = self.centerness_target(
                acid_pos_anchors, acid_pos_bbox_targets)
            acid_pos_decode_bbox_pred = self.bbox_coder.decode(
                acid_pos_anchors, acid_pos_bbox_pred)
            acid_pos_decode_bbox_targets = self.bbox_coder.decode(
                acid_pos_anchors, acid_pos_bbox_targets)

            # regression loss
            acid_loss_bbox = self.loss_bbox(
                acid_pos_decode_bbox_pred,
                acid_pos_decode_bbox_targets,
                weight = acid_centerness_targets,
                avg_factor = 1.0)

            # centerness loss
            acid_loss_centerness = self.loss_centerness(
                acid_pos_centerness,
                acid_centerness_targets,
                avg_factor = num_total_samples)

        else:
            acid_loss_bbox = acid_bbox_pred.sum() * 0
            acid_loss_centerness = acid_centerness.sum() * 0
            acid_centerness_targets = acid_bbox_targets.new_tensor(0.)

        # iodine
        iodine_cls_score = iodine_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        iodine_bbox_pred = iodine_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        iodine_centerness = iodine_centerness.permute(0, 2, 3, 1).reshape(-1)
        iodine_offset = iodine_offset.permute(0, 2, 3, 1).reshape(-1, 2)
        iodine_bbox_targets = iodine_bbox_targets.reshape(-1, 4)
        iodine_labels = iodine_labels.reshape(-1)
        iodine_label_weights = iodine_label_weights.reshape(-1)
        iodine_offsets = iodine_offsets.reshape(-1, 2)

        # classification loss
        iodine_loss_cls = self.loss_cls(
            iodine_cls_score, iodine_labels, iodine_label_weights, avg_factor = num_total_samples)

        # offset loss
        stride = abs(anchors[1, 0] - anchors[0, 0])
        iodine_loss_offset = self.loss_offset(iodine_offset, iodine_offsets / stride)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        iodine_pos_inds = ((iodine_labels >= 0)
                           & (iodine_labels < bg_class_ind)).nonzero().squeeze(1)

        if len(iodine_pos_inds) > 0:
            iodine_pos_bbox_targets = iodine_bbox_targets[iodine_pos_inds]
            iodine_pos_bbox_pred = iodine_bbox_pred[iodine_pos_inds]
            iodine_pos_anchors = anchors[iodine_pos_inds]
            iodine_pos_centerness = iodine_centerness[iodine_pos_inds]

            iodine_centerness_targets = self.centerness_target(
                iodine_pos_anchors, iodine_pos_bbox_targets)
            iodine_pos_decode_bbox_pred = self.bbox_coder.decode(
                iodine_pos_anchors, iodine_pos_bbox_pred)
            iodine_pos_decode_bbox_targets = self.bbox_coder.decode(
                iodine_pos_anchors, iodine_pos_bbox_targets)

            # regression loss
            iodine_loss_bbox = self.loss_bbox(
                iodine_pos_decode_bbox_pred,
                iodine_pos_decode_bbox_targets,
                weight = iodine_centerness_targets,
                avg_factor = 1.0)

            # centerness loss
            iodine_loss_centerness = self.loss_centerness(
                iodine_pos_centerness,
                iodine_centerness_targets,
                avg_factor = num_total_samples)

        else:
            iodine_loss_bbox = iodine_bbox_pred.sum() * 0
            iodine_loss_centerness = iodine_centerness.sum() * 0
            iodine_centerness_targets = iodine_bbox_targets.new_tensor(0.)

        return (acid_loss_cls, acid_loss_bbox, acid_loss_centerness, acid_loss_offset, acid_centerness_targets.sum(),
                iodine_loss_cls, iodine_loss_bbox, iodine_loss_centerness, iodine_loss_offset, iodine_centerness_targets.sum())

    @force_fp32(apply_to = ('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             acid_cls_scores, iodine_cls_scores,
             acid_bbox_preds, iodine_bbox_preds,
             acid_centernesses, iodine_centernesses,
             acid_offset, iodine_offset,
             img_metas,
             acid_gt_bboxes, iodine_gt_bboxes,
             acid_gt_labels, iodine_gt_labels,
             acid_gt_bboxes_ignore = None,
             iodine_gt_bboxes_ignore = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, num_anchors * 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in acid_cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = acid_cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device = device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            acid_gt_bboxes,
            iodine_gt_bboxes,
            img_metas,
            acid_gt_bboxes_ignore,
            iodine_gt_bboxes_ignore,
            acid_gt_labels,
            iodine_gt_labels,
            label_channels = label_channels)
        if cls_reg_targets is None:
            return None

        (anchors_list, acid_labels_list, acid_label_weights_list, acid_bbox_targets_list, acid_bbox_weights_list, acid_offset_list,
         iodine_labels_list, iodine_label_weights_list, iodine_bbox_targets_list, iodine_bbox_weights_list, iodine_offset_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype = torch.float,
                         device = device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        (acid_loss_cls, acid_loss_bbox, acid_loss_centerness, acid_loss_offset, acid_bbox_avg_factor,
         iodine_loss_cls, iodine_loss_bbox, iodine_loss_centerness, iodine_loss_offset, iodine_bbox_avg_factor) = multi_apply(
            self.loss_single,
            anchor_list,
            acid_cls_scores, iodine_cls_scores,
            acid_bbox_preds, iodine_bbox_preds,
            acid_centernesses, iodine_centernesses,
            acid_offset, iodine_offset,
            acid_labels_list, acid_label_weights_list, acid_bbox_targets_list, acid_offset_list,
            iodine_labels_list, iodine_label_weights_list, iodine_bbox_targets_list, iodine_offset_list,
            num_total_samples = num_total_samples)

        acid_bbox_avg_factor = sum(acid_bbox_avg_factor)
        acid_bbox_avg_factor = reduce_mean(acid_bbox_avg_factor).clamp_(min = 1).item()
        acid_loss_bbox = list(map(lambda x: x / acid_bbox_avg_factor, acid_loss_bbox))
        iodine_bbox_avg_factor = sum(iodine_bbox_avg_factor)
        iodine_bbox_avg_factor = reduce_mean(iodine_bbox_avg_factor).clamp_(min = 1).item()
        iodine_loss_bbox = list(map(lambda x: x / iodine_bbox_avg_factor, iodine_loss_bbox))
        return dict(
            acid_loss_cls = acid_loss_cls,
            acid_loss_bbox = acid_loss_bbox,
            acid_loss_centerness = acid_loss_centerness,
            iodine_loss_cls = iodine_loss_cls,
            iodine_loss_bbox = iodine_loss_bbox,
            iodine_loss_centerness = iodine_loss_centerness)

    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim = 1)
        top_bottom = torch.stack([t_, b_], dim = 1)
        centerness = torch.sqrt(
            (left_right.min(dim = -1)[0] / left_right.max(dim = -1)[0]) *
            (top_bottom.min(dim = -1)[0] / top_bottom.max(dim = -1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    @force_fp32(apply_to = ('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg = None,
                   rescale = False,
                   with_nms = True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_anchors * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device = device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        img_shapes = [
            img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_anchors,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale = False,
                    with_nms = True):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (N, num_anchors * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_anchors * 1, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device = device, dtype = torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)

            # Always keep topk op for dynamic input in onnx
            if nms_pre_tensor > 0 and (torch.onnx.is_in_onnx_export()
                                       or scores.shape[-2] > nms_pre_tensor):
                from torch import _shape_as_tensor
                # keep shape as tensor and get k
                num_anchor = _shape_as_tensor(scores)[-2].to(device)
                nms_pre = torch.where(nms_pre_tensor < num_anchor,
                                      nms_pre_tensor, num_anchor)

                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]
            else:
                anchors = anchors.expand_as(bbox_pred)

            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape = img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim = 1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                scale_factors).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim = 1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim = 1)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            batch_mlvl_scores, _ = (
                    batch_mlvl_scores *
                    batch_mlvl_centerness.unsqueeze(2).expand_as(batch_mlvl_scores)
            ).max(-1)
            _, topk_inds = batch_mlvl_scores.topk(deploy_nms_pre)
            batch_inds = torch.arange(batch_size).view(-1,
                                                       1).expand_as(topk_inds)
            batch_mlvl_scores = batch_mlvl_scores[batch_inds, topk_inds, :]
            batch_mlvl_bboxes = batch_mlvl_bboxes[batch_inds, topk_inds, :]
            batch_mlvl_centerness = batch_mlvl_centerness[batch_inds,
                                                          topk_inds]
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim = -1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors = mlvl_centerness)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    acid_gt_bboxes_list,
                    iodine_gt_bboxes_list,
                    img_metas,
                    acid_gt_bboxes_ignore_list = None,
                    iodine_gt_bboxes_ignore_list = None,
                    acid_gt_labels_list = None,
                    iodine_gt_labels_list = None,
                    label_channels = 1,
                    unmap_outputs = True):
        """Get targets for ATSS head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if acid_gt_bboxes_ignore_list is None:
            acid_gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
            iodine_gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if acid_gt_labels_list is None:
            acid_gt_labels_list = [None for _ in range(num_imgs)]
            iodine_gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_acid_labels, all_acid_label_weights, all_acid_bbox_targets, all_acid_bbox_weights, all_acid_offsets,
         all_iodine_labels, all_iodine_label_weights, all_iodine_bbox_targets, all_iodine_bbox_weights, all_iodine_offsets,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            acid_gt_bboxes_list,
            iodine_gt_bboxes_list,
            acid_gt_bboxes_ignore_list,
            iodine_gt_bboxes_ignore_list,
            acid_gt_labels_list,
            iodine_gt_labels_list,
            img_metas,
            label_channels = label_channels,
            unmap_outputs = unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_acid_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        acid_labels_list = images_to_levels(all_acid_labels, num_level_anchors)
        acid_label_weights_list = images_to_levels(all_acid_label_weights, num_level_anchors)
        acid_bbox_targets_list = images_to_levels(all_acid_bbox_targets, num_level_anchors)
        acid_bbox_weights_list = images_to_levels(all_acid_bbox_weights, num_level_anchors)
        acid_offset_list = images_to_levels(all_acid_offsets, num_level_anchors)
        iodine_labels_list = images_to_levels(all_iodine_labels, num_level_anchors)
        iodine_label_weights_list = images_to_levels(all_iodine_label_weights, num_level_anchors)
        iodine_bbox_targets_list = images_to_levels(all_iodine_bbox_targets, num_level_anchors)
        iodine_bbox_weights_list = images_to_levels(all_iodine_bbox_weights, num_level_anchors)
        iodine_offset_list = images_to_levels(all_iodine_offsets, num_level_anchors)
        return (anchors_list, acid_labels_list, acid_label_weights_list, acid_bbox_targets_list, acid_bbox_weights_list, acid_offset_list,
                iodine_labels_list, iodine_label_weights_list, iodine_bbox_targets_list, iodine_bbox_weights_list, iodine_offset_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           acid_gt_bboxes, iodine_gt_bboxes,
                           acid_gt_bboxes_ignore, iodine_gt_bboxes_ignore,
                           acid_gt_labels, iodine_gt_labels,
                           img_meta,
                           label_channels = 1,
                           unmap_outputs = True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4)
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        # offset
        assert len(acid_gt_bboxes) == len(
            iodine_gt_bboxes), f'number of acid and iodine gt bboxes must match, file name: {img_meta["filename"]}'
        acid_gt_bboxes = acid_gt_bboxes[acid_gt_bboxes[:, 0].sort()[1]]
        iodine_gt_bboxes = iodine_gt_bboxes[iodine_gt_bboxes[:, 0].sort()[1]]
        acid_iodine_offset = iodine_gt_bboxes - acid_gt_bboxes
        acid_iodine_offset = (acid_iodine_offset[:, :2] + acid_iodine_offset[:, 2:]) / 2
        acid_iodine_offset = torch.stack([acid_iodine_offset[:, 1], acid_iodine_offset[:, 0]], dim = 1)

        # acid
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             acid_gt_bboxes, acid_gt_bboxes_ignore,
                                             acid_gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              acid_gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        acid_bbox_targets = torch.zeros_like(anchors)
        acid_bbox_weights = torch.zeros_like(anchors)
        acid_labels = anchors.new_full((num_valid_anchors,),
                                       self.num_classes,
                                       dtype = torch.long)
        acid_label_weights = anchors.new_zeros(num_valid_anchors, dtype = torch.float)
        acid_offsets = anchors.new_zeros((num_valid_anchors, 2), dtype = torch.float)
        acid_offsets[:, :] = torch.mean(acid_iodine_offset, dim = 0, keepdim = True)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            acid_offsets[pos_inds, :] = acid_iodine_offset[sampling_result.info['pos_assigned_gt_inds'], :]
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            acid_bbox_targets[pos_inds, :] = pos_bbox_targets
            acid_bbox_weights[pos_inds, :] = 1.0
            if acid_gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                acid_labels[pos_inds] = 0
            else:
                acid_labels[pos_inds] = acid_gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                acid_label_weights[pos_inds] = 1.0
            else:
                acid_label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            acid_label_weights[neg_inds] = 1.0

        # iodine
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             iodine_gt_bboxes, iodine_gt_bboxes_ignore,
                                             iodine_gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              iodine_gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        iodine_bbox_targets = torch.zeros_like(anchors)
        iodine_bbox_weights = torch.zeros_like(anchors)
        iodine_labels = anchors.new_full((num_valid_anchors,),
                                         self.num_classes,
                                         dtype = torch.long)
        iodine_label_weights = anchors.new_zeros(num_valid_anchors, dtype = torch.float)
        iodine_offsets = anchors.new_zeros((num_valid_anchors, 2), dtype = torch.float)
        iodine_offsets[:, :] = torch.mean(-acid_iodine_offset, dim = 0, keepdim = True)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            iodine_offsets[pos_inds, :] = -acid_iodine_offset[sampling_result.info['pos_assigned_gt_inds'], :]
            if hasattr(self, 'bbox_coder'):
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                # used in VFNetHead
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            iodine_bbox_targets[pos_inds, :] = pos_bbox_targets
            iodine_bbox_weights[pos_inds, :] = 1.0
            if iodine_gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                iodine_labels[pos_inds] = 0
            else:
                iodine_labels[pos_inds] = iodine_gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                iodine_label_weights[pos_inds] = 1.0
            else:
                iodine_label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            iodine_label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            acid_labels = unmap(
                acid_labels, num_total_anchors, inside_flags, fill = self.num_classes)
            acid_label_weights = unmap(acid_label_weights, num_total_anchors,
                                       inside_flags)
            acid_bbox_targets = unmap(acid_bbox_targets, num_total_anchors, inside_flags)
            acid_bbox_weights = unmap(acid_bbox_weights, num_total_anchors, inside_flags)
            acid_offsets = unmap(acid_offsets, num_total_anchors, inside_flags)

            iodine_labels = unmap(
                iodine_labels, num_total_anchors, inside_flags, fill = self.num_classes)
            iodine_label_weights = unmap(iodine_label_weights, num_total_anchors,
                                         inside_flags)
            iodine_bbox_targets = unmap(iodine_bbox_targets, num_total_anchors, inside_flags)
            iodine_bbox_weights = unmap(iodine_bbox_weights, num_total_anchors, inside_flags)
            iodine_offsets = unmap(iodine_offsets, num_total_anchors, inside_flags)

        return (anchors, acid_labels, acid_label_weights, acid_bbox_targets, acid_bbox_weights, acid_offsets,
                iodine_labels, iodine_label_weights, iodine_bbox_targets, iodine_bbox_weights, iodine_offsets,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside

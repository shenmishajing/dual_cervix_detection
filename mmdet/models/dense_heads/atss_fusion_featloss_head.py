import torch
from torch import nn
from ..builder import HEADS, build_head, build_roi_extractor
from .atss_head import ATSSHead
import mmcv
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import numpy as np
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss

@HEADS.register_module()
class ATSSFusionFeatlossHead(ATSSHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(type='CrossEntropyLoss',use_sigmoid=True,loss_weight=1.0),
                 loss_offset=dict(type='SsimLoss', loss_weight=1.0),
                 init_cfg=dict(type='Normal',layer='Conv2d',std=0.01,override=dict(type='Normal',name='atss_cls',std=0.01,bias_prob=0.01)),
                 enlarge=False,
                 **kwargs):
        super(ATSSFusionFeatlossHead, self).__init__(num_classes=num_classes,
                 in_channels=in_channels,
                 stacked_convs=stacked_convs,
                 conv_cfg=conv_cfg,
                 norm_cfg=norm_cfg,
                 loss_centerness=loss_centerness,
                 init_cfg=init_cfg,
                 **kwargs)
        self.enlarge = enlarge
        self.loss_offset = build_loss(loss_offset)
        # self.offset_modules = nn.ModuleList([
        #     nn.Linear(2 * self.bbox_head.conv_out_channels * self.bbox_head.roi_feat_area, self.bbox_head.conv_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
        #     nn.ReLU(),
        #     nn.Linear(self.bbox_head.conv_out_channels, 2), #rescale
        # ])
        self.offset_modules = nn.ModuleList([
            nn.Linear(2*256*7*7,256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # rescale
        ])

        # # fusion b
        # self.fusion_modules_b = nn.ModuleList([
        #     nn.Conv2d(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels,3,stride=1, padding=1),
        #     nn.ReLU()
        # ])
        # fusion b
        self.fusion_modules_b = nn.ModuleList([
            nn.Conv2d(2 * 256, 256, 3, stride=1, padding=1),
            nn.ReLU()
        ])

    def fusion_feature(self, prim_feats, aux_feats):
        # # fusion B
        feats = torch.cat([prim_feats, aux_feats], dim=1)
        for m in self.fusion_modules_b:
            feats = m(feats)
        return feats

    def _offset_forward(self, feats):

        feats = nn.AdaptiveAvgPool2d(7)(feats)  #torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
        feats = feats.flatten(1)
        for m in self.offset_modules:
            feats = m(feats)
        return feats

    def fusion_befor(self,acid_feats, iodine_feats,img_metas):
        x = []
        x_acid = []
        x_iodine = []
        for lev in range(len(acid_feats)):
            acid_offsets = self._offset_forward(torch.cat([acid_feats[lev], iodine_feats[lev]], dim=1))  ### for fusion 1 and fusion 2
            acid_offsets = acid_offsets * acid_offsets.new_tensor(acid_feats[lev].size()[2:])  ### only for fusion 2
            if acid_feats[0].size()[0] ==2:
                acid_feats_lev_0 = acid_feats[lev][0][:,
                                   int(acid_offsets[0][0]):int(acid_feats[lev].size()[2] + acid_offsets[0][0]),
                                   int(acid_offsets[0][1]):int(acid_feats[lev].size()[3] + acid_offsets[0][1])]
                acid_feats_lev_1 = acid_feats[lev][1][:,
                                   int(acid_offsets[1][0]):int(acid_feats[lev].size()[2] + acid_offsets[1][0]),
                                   int(acid_offsets[1][1]):int(acid_feats[lev].size()[3] + acid_offsets[1][1])]
                # 裁出的图size 为0 时用原图替代
                if acid_feats_lev_0.size()[1]==0 or acid_feats_lev_0.size()[2]==0:
                    acid_feats_lev_0 = acid_feats[lev][0]
                if acid_feats_lev_1.size()[1]==0 or acid_feats_lev_1.size()[2]==0:
                    acid_feats_lev_1 = acid_feats[lev][1]

                acid_feats_lev_0 = np.resize(acid_feats_lev_0.clone().detach().cpu().numpy(),
                                             tuple(iodine_feats[lev][0].size()))
                acid_feats_lev_1 = np.resize(acid_feats_lev_1.clone().detach().cpu().numpy(),
                                             tuple(iodine_feats[lev][0].size()))

                acid_feats_lev_0 = torch.tensor(np.expand_dims(acid_feats_lev_0, axis=0), requires_grad=True).cuda()
                acid_feats_lev_1 = torch.tensor(np.expand_dims(acid_feats_lev_1, axis=0), requires_grad=True).cuda()
                acid_feats_lev = torch.cat([acid_feats_lev_0, acid_feats_lev_1], dim=0)
            elif acid_feats[0].size()[0] ==1:
                acid_feats_lev_0 = acid_feats[lev][0][:,
                                   int(acid_offsets[0][0]):int(acid_feats[lev].size()[2] + acid_offsets[0][0]),
                                   int(acid_offsets[0][1]):int(acid_feats[lev].size()[3] + acid_offsets[0][1])]
                # 裁出的图size 为0 时用原图替代
                if acid_feats_lev_0.size()[1]==0 or acid_feats_lev_0.size()[2]==0:
                    acid_feats_lev_0 = acid_feats[lev][0]

                acid_feats_lev_0 = np.resize(acid_feats_lev_0.clone().detach().cpu().numpy(),
                                             tuple(iodine_feats[lev][0].size()))
                acid_feats_lev_0 = torch.tensor(np.expand_dims(acid_feats_lev_0, axis=0), requires_grad=True).cuda()
                acid_feats_lev = acid_feats_lev_0
            else:
                print('error dimention')

            x.append(self.fusion_feature(acid_feats_lev, iodine_feats[lev]))
            x_acid.append(acid_feats_lev)
            x_iodine.append(iodine_feats[lev])

        x = tuple(x)
        return x,tuple(x_acid),iodine_feats



    def forward_train(self,
                      acid_feats, iodine_feats,
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
        x, x_acid, x_iodine = self.fusion_befor(acid_feats, iodine_feats, img_metas)

        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, x_acid, x_iodine, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list



    def loss_single(self, anchors, cls_score, bbox_pred, centerness, feats_acid, feats_iodine, labels,
                    label_weights, bbox_targets, num_total_samples):
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
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)


        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)

            # centerness loss
            loss_centerness = self.loss_centerness(
                pos_centerness,
                centerness_targets,
                avg_factor=num_total_samples)

            # feature transform loss
            feats_acid = feats_acid.permute(0, 2, 3, 1).reshape(cls_score.size()[0], -1)
            feats_iodine = feats_iodine.permute(0, 2, 3, 1).reshape(cls_score.size()[0], -1)
            loss_offset = self.loss_offset(
                feats_acid[pos_inds],
                feats_iodine[pos_inds],
                centerness_targets,
                avg_factor=1.0,
                reduction_override=None)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

            # feature transform loss
            loss_offset = (feats_acid[pos_inds] - feats_iodine[pos_inds]).sum() * 0

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum(), loss_offset

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             feats_acid, feats_iodine,
             gt_bboxes_ignore=None):
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
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness,\
            bbox_avg_factor,loss_offset = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                feats_acid,
                feats_iodine,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        loss_offset = list(map(lambda x: x / bbox_avg_factor, loss_offset))
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness,loss_offset=loss_offset)


    def simple_test(self, acid_feats, iodine_feats, img_metas, rescale=False):
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
        return self.simple_test_bboxes(acid_feats, iodine_feats, img_metas, rescale=rescale)


    def simple_test_bboxes(self, acid_feats, iodine_feats, img_metas, rescale=False):
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

        feats, _, _ = self.fusion_befor(acid_feats, iodine_feats, img_metas)

        outs = self(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list
import torch
from mmcv.runner import force_fp32
import numpy as np
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .convfc_bbox_head import Shared2FCBBoxHead
import itertools
import math
from torch import nn

@HEADS.register_module()
class DualConsisLossbfcBBoxHead(Shared2FCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self, loss_conscore = dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_confeat = dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_confeat_tot = dict(type = 'L1Loss', loss_weight = 1.0),loss_progcls=dict(type='L1Loss', loss_weight=1.0),
                 loss_progbox=dict(type='L1Loss', loss_weight=1.0),
                 *args, **kwargs):
        super(DualConsisLossbfcBBoxHead, self).__init__(*args, **kwargs)
        self.loss_conscore = build_loss(loss_conscore)
        self.loss_confeat = build_loss(loss_confeat)
        self.loss_confeat_tot = build_loss(loss_confeat_tot)
        self.loss_progcls = build_loss(loss_progcls)
        self.loss_progbox = build_loss(loss_progbox)




        self.hidden_dim = 256 #self.bbox_head.conv_out_channels  # 256
        self.dim_dynamic = 49 #self.bbox_head.roi_feat_area  # 49

        self.dynamic_layer = nn.Linear(self.hidden_dim *self.dim_dynamic,self.dim_dynamic)  # 256 to (2*256)   ,49 can be obtain by cat
        self.norm1 = nn.LayerNorm(self.dim_dynamic)  # 49
        self.activation = nn.ReLU(inplace=True)




    def forward(self, acid_bbox_feats):
        acid_cls_score, acid_bbox_pred = super(DualConsisLossbfcBBoxHead, self).forward(acid_bbox_feats)
        return acid_cls_score, acid_bbox_pred

    def _get_target_single(self,
                           acid_pos_bboxes,
                           acid_neg_bboxes,
                           acid_pos_gt_bboxes,
                           acid_pos_gt_labels,
                           acid_gt_bboxes,
                           acid_pos_assigned_gt_inds,
                           img_meta, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        # offset
        #assert len(acid_gt_bboxes) == len(iodine_gt_bboxes), 'number of acid and iodine gt bboxes must match'
        acid_gt_bboxes_inds = acid_gt_bboxes[:, 0].sort()[1]
        acid_gt_bboxes_sorted = acid_gt_bboxes[acid_gt_bboxes_inds]
        #iodine_gt_bboxes_inds = iodine_gt_bboxes[:, 0].sort()[1]
        #iodine_gt_bboxes_sorted = iodine_gt_bboxes[iodine_gt_bboxes_inds]

        # acid
        acid_num_pos = acid_pos_bboxes.size(0)
        acid_num_neg = acid_neg_bboxes.size(0)
        acid_num_samples = acid_num_pos + acid_num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        acid_labels = acid_pos_bboxes.new_full((acid_num_samples,), self.num_classes, dtype = torch.long)
        acid_label_weights = acid_pos_bboxes.new_zeros(acid_num_samples)
        acid_bbox_targets = acid_pos_bboxes.new_zeros(acid_num_samples, 4)
        acid_bbox_weights = acid_pos_bboxes.new_zeros(acid_num_samples, 4)
        #acid_offset_targets = acid_pos_bboxes.new_zeros(acid_num_samples, 2)
        acid_offset_weights = acid_pos_bboxes.new_zeros(acid_num_samples, 1)
        if acid_num_pos > 0:
            acid_labels[:acid_num_pos] = acid_pos_gt_labels
            acid_pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            acid_label_weights[:acid_num_pos] = acid_pos_weight
            if not self.reg_decoded_bbox:
                acid_pos_bbox_targets = self.bbox_coder.encode(
                    acid_pos_bboxes, acid_pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                acid_pos_bbox_targets = acid_pos_gt_bboxes
            acid_bbox_targets[:acid_num_pos, :] = acid_pos_bbox_targets
            acid_bbox_weights[:acid_num_pos, :] = 1
            # for i in range(acid_num_pos):
            #     acid_offset_target = torch.mean(iodine_gt_bboxes_sorted[acid_gt_bboxes_inds == acid_pos_assigned_gt_inds[i]], dim = 0)
            #     acid_offset_target = acid_offset_target - acid_pos_bboxes[i, :]
            #     acid_offset_target = (acid_offset_target[:2] + acid_offset_target[2:]) / 2
            #     acid_offset_target = acid_offset_target / acid_offset_target.new_tensor(img_meta['pad_shape'][:2])
            #     acid_offset_targets[i, :] = acid_offset_target
            acid_offset_weights[:acid_num_pos, :] = 1
        if acid_num_neg > 0:
            acid_label_weights[-acid_num_neg:] = 1.0


        return (acid_labels,
                acid_label_weights,
                acid_bbox_targets,
                acid_bbox_weights,
                #acid_offset_targets,
                acid_offset_weights)

    def get_targets(self,
                    acid_sampling_results,
                    acid_gt_bboxes,
                    acid_gt_labels,
                    img_metas, rcnn_train_cfg,
                    concat = True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        acid_pos_bboxes_list = [res.pos_bboxes for res in acid_sampling_results]
        acid_neg_bboxes_list = [res.neg_bboxes for res in acid_sampling_results]
        acid_pos_gt_bboxes_list = [res.pos_gt_bboxes for res in acid_sampling_results]
        acid_pos_gt_labels_list = [res.pos_gt_labels for res in acid_sampling_results]
        acid_pos_assigned_gt_inds_list = [res.info['pos_assigned_gt_inds'] for res in acid_sampling_results]


        bbox_targets = multi_apply(
            self._get_target_single,
            acid_pos_bboxes_list,
            acid_neg_bboxes_list,
            acid_pos_gt_bboxes_list,
            acid_pos_gt_labels_list,
            acid_gt_bboxes,
            acid_pos_assigned_gt_inds_list,
            img_metas, cfg = rcnn_train_cfg)

        if concat:
            bbox_targets = [torch.cat(b) for b in bbox_targets]
        return bbox_targets

    def iou(self, bbox1, bbox2):
        # bbox1 = [x0,y0,x1,y1]
        x0, y0, x1, y1 = bbox1
        x2, y2, x3, y3 = bbox2
        s1 = (x1 - x0) * (y1 - y0)
        s2 = (x3 - x2) * (y3 - y2)
        w = max(0, min(x1, x3) - max(x0, x2))
        h = max(0, min(y1, y3) - max(y0, y2))
        inter = w * h
        iou = inter / (s1 + s2 - inter + 1e-6)
        return iou

    @force_fp32(apply_to = ('cls_score', 'bbox_pred'))
    def loss(self,
             acid_cls_score,iodine_cls_score,
             acid_bbox_pred,
             acid_bbox_feats, iodine_bbox_feats,
             acid_fusion_cls_score,
             acid_fusion_bbox_pred,
             acid_rois,
             acid_labels,
             acid_label_weights,
             acid_bbox_targets,
             acid_bbox_weights,
             #acid_offset_targets,
             acid_offset_weights,
             reduction_override = None):
        losses = dict()

        # acid
        if acid_cls_score is not None:
            avg_factor = max(torch.sum(acid_label_weights > 0).float().item(), 1.)
            if acid_cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    acid_cls_score,
                    acid_labels,
                    acid_label_weights,
                    avg_factor = avg_factor,
                    reduction_override = reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['acid_loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(acid_cls_score, acid_labels)
                    losses.update(acc_)
                else:
                    losses['acid_acc'] = accuracy(acid_cls_score, acid_labels)
        if acid_bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            acid_pos_inds = (acid_labels >= 0) & (acid_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if acid_pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    acid_bbox_pred = self.bbox_coder.decode(acid_rois[:, 1:], acid_bbox_pred)
                if self.reg_class_agnostic:
                    acid_pos_bbox_pred = acid_bbox_pred.view(
                        acid_bbox_pred.size(0), 4)[acid_pos_inds.type(torch.bool)]
                else:
                    acid_pos_bbox_pred = acid_bbox_pred.view(
                        acid_bbox_pred.size(0), -1,
                        4)[acid_pos_inds.type(torch.bool),
                           acid_labels[acid_pos_inds.type(torch.bool)]]
                losses['acid_loss_bbox'] = self.loss_bbox(
                    acid_pos_bbox_pred,
                    acid_bbox_targets[acid_pos_inds.type(torch.bool)],
                    acid_bbox_weights[acid_pos_inds.type(torch.bool)],
                    avg_factor = acid_bbox_targets.size(0),
                    reduction_override = reduction_override)
            else:
                losses['acid_loss_bbox'] = acid_bbox_pred[acid_pos_inds].sum()


        if iodine_cls_score is not None and acid_cls_score is not None:
            avg_factor = max(torch.sum(acid_label_weights > 0).float().item(), 1.)
            if acid_cls_score.numel() > 0 and iodine_cls_score.numel() > 0:
                loss_cls_ = self.loss_conscore(
                    acid_cls_score,
                    iodine_cls_score,
                    torch.cat((acid_label_weights.unsqueeze(1), acid_label_weights.unsqueeze(1)), 1),
                    avg_factor = avg_factor,
                    reduction_override = reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['acid_loss_conscore'] = loss_cls_

        ##先计算单个模态内两两特征之间的相似性之和，再计算两个模态相似性的差异作为损失，该种计算方式训练后损失越来越大，不可取
        # if acid_bbox_feats is not None and iodine_bbox_feats is not None:
        #     bg_class_ind = self.num_classes
        #     # 0~self.num_classes-1 are FG, self.num_classes is BG
        #     acid_pos_inds = (acid_labels >= 0) & (acid_labels < bg_class_ind)
        #     # do not perform bounding box regression for BG anymore.
        #     if acid_pos_inds.any():
        #         acid_bbox_feats_a = acid_bbox_feats[acid_pos_inds.type(torch.bool)].view(
        #             (acid_bbox_feats[acid_pos_inds.type(torch.bool)].size()[0], -1))
        #         iodine_bbox_feats_a = iodine_bbox_feats[acid_pos_inds.type(torch.bool)].view(
        #             (acid_bbox_feats[acid_pos_inds.type(torch.bool)].size()[0], -1))
        #         simil_acid = []
        #         simil_iodine = []
        #         for item in itertools.product(list(range(acid_bbox_feats_a.shape[0])), list(range(acid_bbox_feats_a.shape[0]))):
        #             simil_acid.append(self.loss_confeat(
        #                 acid_bbox_feats_a[item[0]].unsqueeze(0),
        #                 acid_bbox_feats_a[item[1]].unsqueeze(0),
        #                 acid_offset_weights[acid_pos_inds.type(torch.bool)][item[0]].unsqueeze(0),
        #                 avg_factor=1, #acid_bbox_feats_a.size(0)
        #                 reduction_override=reduction_override).cpu().detach().numpy())
        #             simil_iodine.append(self.loss_confeat(
        #                 iodine_bbox_feats_a[item[0]].unsqueeze(0),
        #                 iodine_bbox_feats_a[item[1]].unsqueeze(0),
        #                 acid_offset_weights[acid_pos_inds.type(torch.bool)][item[0]].unsqueeze(0),
        #                 avg_factor=1, #iodine_bbox_feats_a.size(0),
        #                 reduction_override=reduction_override).cpu().detach().numpy())
        #         simil_acid = torch.from_numpy(np.array(simil_acid)).cuda()
        #         simil_iodine = torch.from_numpy(np.array(simil_iodine)).cuda()
        #         simil_acid.requires_grad = True
        #         simil_iodine.requires_grad = True
        #         del acid_bbox_feats_a,iodine_bbox_feats_a
        #
        #         losses['acid_loss_confeat'] = self.loss_confeat_tot(
        #             simil_acid.mean(),
        #             simil_iodine.mean(),
        #             acid_offset_weights[acid_pos_inds.type(torch.bool)][0].unsqueeze(0),
        #             avg_factor=1,  # (acid_bbox_feats_a.size(0)) * (iodine_bbox_feats_a.size(0)),
        #             reduction_override=reduction_override)
        #
        #         del simil_acid, simil_iodine
        #     else:
        #         losses['acid_loss_confeat'] = 0 * (acid_bbox_feats[acid_pos_inds]-iodine_bbox_feats[acid_pos_inds]).sum()



        if acid_bbox_feats is not None and iodine_bbox_feats is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            acid_pos_inds = (acid_labels >= 0) & (acid_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if acid_pos_inds.any():
                acid_bbox_feats_a = acid_bbox_feats[acid_pos_inds.type(torch.bool)].view(
                    (acid_bbox_feats[acid_pos_inds.type(torch.bool)].size()[0], -1))
                iodine_bbox_feats_a = iodine_bbox_feats[acid_pos_inds.type(torch.bool)].view(
                    (acid_bbox_feats[acid_pos_inds.type(torch.bool)].size()[0], -1))

                acid_bbox_feats_a = self.activation(self.norm1(self.dynamic_layer(acid_bbox_feats_a)))
                iodine_bbox_feats_a = self.activation(self.norm1(self.dynamic_layer(iodine_bbox_feats_a)))


                dis_feat = []
                for item in range(acid_bbox_feats_a.shape[0]):
                    dis_feat.append(self.loss_confeat_tot(
                        acid_bbox_feats_a[item].unsqueeze(0),
                        iodine_bbox_feats_a[item].unsqueeze(0),
                        acid_offset_weights[acid_pos_inds.type(torch.bool)][item].unsqueeze(0),
                        avg_factor=1, #acid_bbox_feats_a.size(0)
                        reduction_override=reduction_override).cpu().detach().numpy()) #/acid_bbox_feats_a.size()[1] ???

                dis_feat = torch.from_numpy(np.array(dis_feat)).cuda()
                dis_feat.requires_grad = True
                del acid_bbox_feats_a,iodine_bbox_feats_a

                losses['acid_loss_confeat'] = dis_feat.mean()

                del dis_feat
            else:
                losses['acid_loss_confeat'] = 0 * (acid_bbox_feats[acid_pos_inds]-iodine_bbox_feats[acid_pos_inds]).sum()








        # #progressive loss cls
        # if acid_fusion_cls_score is not None and acid_cls_score is not None:
        #     avg_factor = max(torch.sum(acid_label_weights > 0).float().item(), 1.)
        #     if acid_cls_score.numel() > 0 and acid_fusion_cls_score.numel() > 0:
        #         loss_cls_ = self.loss_progcls(
        #             acid_cls_score,
        #             acid_fusion_cls_score-0.2,
        #             torch.cat((acid_label_weights.unsqueeze(1), acid_label_weights.unsqueeze(1)), 1),
        #             avg_factor = avg_factor,
        #             reduction_override = reduction_override)
        #         if isinstance(loss_cls_, dict):
        #             losses.update(loss_cls_)
        #         else:
        #             losses['acid_loss_progcls'] = loss_cls_
        # ### progressive loss bbox
        # if acid_bbox_pred is not None and acid_fusion_bbox_pred is not None:
        #     bg_class_ind = self.num_classes
        #     # 0~self.num_classes-1 are FG, self.num_classes is BG
        #     acid_pos_inds = (acid_labels >= 0) & (acid_labels < bg_class_ind)
        #     # do not perform bounding box regression for BG anymore.
        #     if acid_pos_inds.any():
        #         if True: #self.reg_decoded_bbox:
        #             # When the regression loss (e.g. `IouLoss`,
        #             # `GIouLoss`, `DIouLoss`) is applied directly on
        #             # the decoded bounding boxes, it decodes the
        #             # already encoded coordinates to absolute format.
        #             acid_bbox_pred = self.bbox_coder.decode(acid_rois[:, 1:], acid_bbox_pred)
        #             acid_fusion_bbox_pred = self.bbox_coder.decode(acid_rois[:, 1:], acid_fusion_bbox_pred)
        #             acid_bbox_targets = self.bbox_coder.decode(acid_rois[:, 1:], acid_bbox_targets)
        #         if self.reg_class_agnostic:
        #             acid_pos_bbox_pred = acid_bbox_pred.view(
        #                 acid_bbox_pred.size(0), 4)[acid_pos_inds.type(torch.bool)]
        #             acid_fusion_pos_bbox_pred = acid_fusion_bbox_pred.view(
        #                 acid_fusion_bbox_pred.size(0), 4)[acid_pos_inds.type(torch.bool)]
        #             acid_pos_bbox_targets = acid_bbox_targets.view(
        #                 acid_bbox_targets.size(0), 4)[acid_pos_inds.type(torch.bool)]
        #         else:
        #             acid_pos_bbox_pred = acid_bbox_pred.view(acid_bbox_pred.size(0), -1,4)[acid_pos_inds.type(torch.bool),
        #                    acid_labels[acid_pos_inds.type(torch.bool)]]
        #             acid_fusion_pos_bbox_pred = acid_fusion_bbox_pred.view(acid_fusion_bbox_pred.size(0), -1,4)[acid_pos_inds.type(torch.bool),
        #                    acid_labels[acid_pos_inds.type(torch.bool)]]
        #             acid_pos_bbox_targets = acid_bbox_targets.view(acid_bbox_targets.size(0), -1, 4)[acid_pos_inds.type(torch.bool),
        #                 acid_labels[acid_pos_inds.type(torch.bool)]]
        #
        #         #compute IOU
        #         IOU = []
        #         IOU_FUS = []
        #         for id in range(acid_pos_bbox_pred.size()[0]):
        #             IOU.append(self.iou(acid_pos_bbox_pred[id],acid_pos_bbox_targets[id]).cpu().detach().numpy())
        #             IOU_FUS.append(self.iou(acid_fusion_pos_bbox_pred[id], acid_pos_bbox_targets[id]).cpu().detach().numpy())
        #
        #         IOU = torch.from_numpy(np.array(IOU)).cuda()
        #         IOU_FUS = torch.from_numpy(np.array(IOU_FUS)).cuda()
        #         IOU.requires_grad = True
        #         IOU_FUS.requires_grad = True
        #         losses['acid_loss_progbox'] = self.loss_progbox(
        #             IOU,
        #             IOU_FUS-0.2,
        #             acid_bbox_weights[acid_pos_inds.type(torch.bool),0].unsqueeze(1),
        #             avg_factor = acid_pos_bbox_targets.size(0),
        #             reduction_override = reduction_override)
        #     else:
        #         losses['acid_loss_progbox'] = 0 * acid_bbox_pred[acid_pos_inds].sum()

        return losses

import torch
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from .convfc_bbox_head import Shared2FCBBoxHead


@HEADS.register_module()
class DualBBoxHead(Shared2FCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self, loss_offset = dict(type = 'SmoothL1Loss', beta = 1.0 / 9.0, loss_weight = 1.0), *args, **kwargs):
        super(DualBBoxHead, self).__init__(*args, **kwargs)
        self.loss_offset = build_loss(loss_offset)

    def forward(self, acid_bbox_feats, iodine_bbox_feats):
        acid_cls_score, acid_bbox_pred = super(DualBBoxHead, self).forward(acid_bbox_feats)
        iodine_cls_score, iodine_bbox_pred = super(DualBBoxHead, self).forward(iodine_bbox_feats)
        return acid_cls_score, iodine_cls_score, acid_bbox_pred, iodine_bbox_pred

    def _get_target_single(self,
                           acid_pos_bboxes, iodine_pos_bboxes,
                           acid_neg_bboxes, iodine_neg_bboxes,
                           acid_pos_gt_bboxes, iodine_pos_gt_bboxes,
                           acid_pos_gt_labels, iodine_pos_gt_labels,
                           acid_gt_bboxes, iodine_gt_bboxes,
                           acid_pos_assigned_gt_inds, iodine_pos_assigned_gt_inds,
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
        assert len(acid_gt_bboxes) == len(iodine_gt_bboxes), 'number of acid and iodine gt bboxes must match'
        acid_gt_bboxes_inds = acid_gt_bboxes[:, 0].sort()[1]
        acid_gt_bboxes_sorted = acid_gt_bboxes[acid_gt_bboxes_inds]
        iodine_gt_bboxes_inds = iodine_gt_bboxes[:, 0].sort()[1]
        iodine_gt_bboxes_sorted = iodine_gt_bboxes[iodine_gt_bboxes_inds]
        acid_iodine_offsets = iodine_gt_bboxes_sorted - acid_gt_bboxes_sorted
        acid_iodine_offsets = (acid_iodine_offsets[:, :2] + acid_iodine_offsets[:, 2:]) / 2
        global_offset_targets = torch.mean(acid_iodine_offsets, dim = 0, keepdim = True)
        global_offset_targets = global_offset_targets / global_offset_targets.new_tensor(img_meta['pad_shape'][:2])

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
        acid_offset_targets = acid_pos_bboxes.new_zeros(acid_num_samples, 2)
        acid_offset_weights = acid_pos_bboxes.new_zeros(acid_num_samples, 2)
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
            for i in range(acid_num_pos):
                acid_offset_target = torch.mean(iodine_gt_bboxes_sorted[acid_gt_bboxes_inds == acid_pos_assigned_gt_inds[i]], dim = 0)
                acid_offset_target = acid_offset_target - acid_pos_bboxes[i, :]
                acid_offset_target = (acid_offset_target[:2] + acid_offset_target[2:]) / 2
                acid_offset_target = acid_offset_target / torch.stack(
                    [acid_pos_bboxes[i, 2] - acid_pos_bboxes[i, 0], acid_pos_bboxes[i, 3] - acid_pos_bboxes[i, 1]])
                acid_offset_targets[i, :] = acid_offset_target
            acid_offset_weights[:acid_num_pos, :] = 1
        if acid_num_neg > 0:
            acid_label_weights[-acid_num_neg:] = 1.0

        # iodine
        iodine_num_pos = iodine_pos_bboxes.size(0)
        iodine_num_neg = iodine_neg_bboxes.size(0)
        iodine_num_samples = iodine_num_pos + iodine_num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        iodine_labels = iodine_pos_bboxes.new_full((iodine_num_samples,), self.num_classes, dtype = torch.long)
        iodine_label_weights = iodine_pos_bboxes.new_zeros(iodine_num_samples)
        iodine_bbox_targets = iodine_pos_bboxes.new_zeros(iodine_num_samples, 4)
        iodine_bbox_weights = iodine_pos_bboxes.new_zeros(iodine_num_samples, 4)
        iodine_offset_targets = iodine_pos_bboxes.new_zeros(iodine_num_samples, 2)
        iodine_offset_weights = iodine_pos_bboxes.new_zeros(iodine_num_samples, 2)
        if iodine_num_pos > 0:
            iodine_labels[:iodine_num_pos] = iodine_pos_gt_labels
            iodine_pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            iodine_label_weights[:iodine_num_pos] = iodine_pos_weight
            if not self.reg_decoded_bbox:
                iodine_pos_bbox_targets = self.bbox_coder.encode(
                    iodine_pos_bboxes, iodine_pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                iodine_pos_bbox_targets = iodine_pos_gt_bboxes
            iodine_bbox_targets[:iodine_num_pos, :] = iodine_pos_bbox_targets
            iodine_bbox_weights[:iodine_num_pos, :] = 1
            for i in range(iodine_num_pos):
                iodine_offset_target = torch.mean(acid_gt_bboxes_sorted[iodine_gt_bboxes_inds == iodine_pos_assigned_gt_inds[i]], dim = 0)
                iodine_offset_target = iodine_offset_target - iodine_pos_bboxes[i, :]
                iodine_offset_target = (iodine_offset_target[:2] + iodine_offset_target[2:]) / 2
                iodine_offset_target = iodine_offset_target / torch.stack(
                    [iodine_pos_bboxes[i, 2] - iodine_pos_bboxes[i, 0], iodine_pos_bboxes[i, 3] - iodine_pos_bboxes[i, 1]])
                iodine_offset_targets[i, :] = iodine_offset_target
            iodine_offset_weights[:iodine_num_pos, :] = 1
        if iodine_num_neg > 0:
            iodine_label_weights[-iodine_num_neg:] = 1.0

        return (acid_labels, iodine_labels,
                acid_label_weights, iodine_label_weights,
                acid_bbox_targets, iodine_bbox_targets,
                acid_bbox_weights, iodine_bbox_weights,
                acid_offset_targets, iodine_offset_targets,
                acid_offset_weights, iodine_offset_weights,
                global_offset_targets)

    def get_targets(self,
                    acid_sampling_results, iodine_sampling_results,
                    acid_gt_bboxes, iodine_gt_bboxes,
                    acid_gt_labels, iodine_gt_labels,
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

        iodine_pos_bboxes_list = [res.pos_bboxes for res in iodine_sampling_results]
        iodine_neg_bboxes_list = [res.neg_bboxes for res in iodine_sampling_results]
        iodine_pos_gt_bboxes_list = [res.pos_gt_bboxes for res in iodine_sampling_results]
        iodine_pos_gt_labels_list = [res.pos_gt_labels for res in iodine_sampling_results]
        iodine_pos_assigned_gt_inds_list = [res.info['pos_assigned_gt_inds'] for res in iodine_sampling_results]
        bbox_targets = multi_apply(
            self._get_target_single,
            acid_pos_bboxes_list, iodine_pos_bboxes_list,
            acid_neg_bboxes_list, iodine_neg_bboxes_list,
            acid_pos_gt_bboxes_list, iodine_pos_gt_bboxes_list,
            acid_pos_gt_labels_list, iodine_pos_gt_labels_list,
            acid_gt_bboxes, iodine_gt_bboxes,
            acid_pos_assigned_gt_inds_list, iodine_pos_assigned_gt_inds_list,
            img_metas, cfg = rcnn_train_cfg)

        if concat:
            bbox_targets = [torch.cat(b) for b in bbox_targets]
        return bbox_targets

    @force_fp32(apply_to = ('cls_score', 'bbox_pred'))
    def loss(self,
             acid_cls_score, iodine_cls_score,
             acid_bbox_pred, iodine_bbox_pred,
             acid_proposal_offsets, iodine_proposal_offsets,
             global_offsets,
             acid_rois, iodine_rois,
             acid_labels, iodine_labels,
             acid_label_weights, iodine_label_weights,
             acid_bbox_targets, iodine_bbox_targets,
             acid_bbox_weights, iodine_bbox_weights,
             acid_offset_targets, iodine_offset_targets,
             acid_offset_weights, iodine_offset_weights,
             global_offset_targets,
             reduction_override = None):
        losses = dict()

        # global offset
        # losses['global_offset_loss'] = self.loss_offset(global_offsets, global_offset_targets)

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
        if acid_proposal_offsets is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            acid_pos_inds = (acid_labels >= 0) & (acid_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if acid_pos_inds.any():
                losses['acid_loss_offset'] = self.loss_bbox(
                    acid_proposal_offsets[acid_pos_inds.type(torch.bool)],
                    acid_offset_targets[acid_pos_inds.type(torch.bool)],
                    acid_offset_weights[acid_pos_inds.type(torch.bool)],
                    avg_factor = acid_offset_targets.size(0),
                    reduction_override = reduction_override)
            else:
                losses['acid_loss_offset'] = acid_proposal_offsets[acid_pos_inds].sum()

        # iodine
        if iodine_cls_score is not None:
            avg_factor = max(torch.sum(iodine_label_weights > 0).float().item(), 1.)
            if iodine_cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    iodine_cls_score,
                    iodine_labels,
                    iodine_label_weights,
                    avg_factor = avg_factor,
                    reduction_override = reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['iodine_loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(iodine_cls_score, iodine_labels)
                    losses.update(acc_)
                else:
                    losses['iodine_acc'] = accuracy(iodine_cls_score, iodine_labels)
        if iodine_bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            iodine_pos_inds = (iodine_labels >= 0) & (iodine_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if iodine_pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    iodine_bbox_pred = self.bbox_coder.decode(iodine_rois[:, 1:], iodine_bbox_pred)
                if self.reg_class_agnostic:
                    iodine_pos_bbox_pred = iodine_bbox_pred.view(
                        iodine_bbox_pred.size(0), 4)[iodine_pos_inds.type(torch.bool)]
                else:
                    iodine_pos_bbox_pred = iodine_bbox_pred.view(
                        iodine_bbox_pred.size(0), -1,
                        4)[iodine_pos_inds.type(torch.bool),
                           iodine_labels[iodine_pos_inds.type(torch.bool)]]
                losses['iodine_loss_bbox'] = self.loss_bbox(
                    iodine_pos_bbox_pred,
                    iodine_bbox_targets[iodine_pos_inds.type(torch.bool)],
                    iodine_bbox_weights[iodine_pos_inds.type(torch.bool)],
                    avg_factor = iodine_bbox_targets.size(0),
                    reduction_override = reduction_override)
            else:
                losses['iodine_loss_bbox'] = iodine_bbox_pred[iodine_pos_inds].sum()
        if iodine_proposal_offsets is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            iodine_pos_inds = (iodine_labels >= 0) & (iodine_labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if iodine_pos_inds.any():
                losses['iodine_loss_offset'] = self.loss_bbox(
                    iodine_proposal_offsets[iodine_pos_inds.type(torch.bool)],
                    iodine_offset_targets[iodine_pos_inds.type(torch.bool)],
                    iodine_offset_weights[iodine_pos_inds.type(torch.bool)],
                    avg_factor = iodine_offset_targets.size(0),
                    reduction_override = reduction_override)
            else:
                losses['iodine_loss_offset'] = iodine_proposal_offsets[iodine_pos_inds].sum()

        return losses

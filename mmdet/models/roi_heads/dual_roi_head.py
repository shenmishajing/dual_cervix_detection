import math
import torch
from torch import nn

from mmdet.core import bbox2result, bbox2roi
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class DualRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 offset_level = 0,
                 offset_generator = dict(
                     r = [50, 100, 200],
                     theta = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
                 ),
                 **kwargs):
        super(DualRoIHead, self).__init__(**kwargs)
        self.offset_level = offset_level
        self.offset_generator = offset_generator
        self.offset_anchors = [
            [0, 0], [50, 0], [100, 0], [200, 0],
            [0, 1], [50, 1], [100, 1], [200, 1],
            [0, 0.5], [25, 0.5], [50, 0.5],
            [0, -0.5], [25, -0.5], [50, -0.5],
        ]
        self._offset_anchors_tensor = None
        self.offset_nums = len(self.offset_anchors)
        self.offset_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.offset_modules = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
                nn.ReLU(),
                nn.Linear(self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
                nn.ReLU(),
                nn.Linear(self.bbox_head.conv_out_channels, 3 * self.offset_nums),
            ]),
            nn.ModuleList([
                nn.Linear(2 * self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
                nn.ReLU(),
                nn.Linear(self.bbox_head.conv_out_channels, self.bbox_head.conv_out_channels),
                nn.ReLU(),
                nn.Linear(self.bbox_head.conv_out_channels, 3 * self.offset_nums),
            ])
        ])

    @property
    def offset_anchors_tensor(self, tensor = None):
        if self._offset_anchors_tensor is None:
            self._offset_anchors_tensor = tensor.new_tensor(self.offset_anchors)
        return self._offset_anchors_tensor

    def offset_apply_delta(self, offset, delta):
        pass

    def forward_train(self,
                      acid_feats, iodine_feats,
                      img_metas,
                      acid_proposal_list, iodine_proposal_list,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if acid_gt_bboxes_ignore is None:
                acid_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            if iodine_gt_bboxes_ignore is None:
                iodine_gt_bboxes_ignore = [None for _ in range(num_imgs)]
            acid_sampling_results = []
            iodine_sampling_results = []
            for i in range(num_imgs):
                acid_assign_result = self.bbox_assigner.assign(
                    acid_proposal_list[i], acid_gt_bboxes[i], acid_gt_bboxes_ignore[i],
                    acid_gt_labels[i])
                acid_sampling_result = self.bbox_sampler.sample(
                    acid_assign_result,
                    acid_proposal_list[i],
                    acid_gt_bboxes[i],
                    acid_gt_labels[i],
                    feats = [lvl_feat[i][None] for lvl_feat in acid_feats])
                acid_sampling_results.append(acid_sampling_result)

                iodine_assign_result = self.bbox_assigner.assign(
                    iodine_proposal_list[i], iodine_gt_bboxes[i], iodine_gt_bboxes_ignore[i],
                    iodine_gt_labels[i])
                iodine_sampling_result = self.bbox_sampler.sample(
                    iodine_assign_result,
                    iodine_proposal_list[i],
                    iodine_gt_bboxes[i],
                    iodine_gt_labels[i],
                    feats = [lvl_feat[i][None] for lvl_feat in iodine_feats])
                iodine_sampling_results.append(iodine_sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(acid_feats, iodine_feats,
                                                    acid_sampling_results, iodine_sampling_results,
                                                    acid_gt_bboxes, iodine_gt_bboxes,
                                                    acid_gt_labels, iodine_gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, acid_feats, iodine_feats, acid_rois, iodine_rois, img_metas):
        """Box head forward function used in both training and testing."""
        # # offset
        self.offset_anchors_tensor(acid_feats[0])
        acid_feat, iodine_feat = acid_feats[self.offset_level], iodine_feats[self.offset_level]
        global_offsets = self._offset_forward(torch.cat([acid_feat, iodine_feat], dim = 1), stage = 0)
        # global_offsets_scaled = []
        # for i in range(len(img_metas)):
        #     global_offsets_scaled.append(global_offsets[i] * global_offsets.new_tensor(img_metas[i]['pad_shape'][:2]))
        # global_offsets_scaled = torch.stack(global_offsets_scaled)
        #
        # # acid
        # acid_offsets_added = torch.cat(
        #     [global_offsets_scaled.new_zeros((global_offsets_scaled.shape[0], 1)), global_offsets_scaled, global_offsets_scaled], dim = 1)
        # acid_iodine_rois = []
        # for i in range(len(acid_offsets_added)):
        #     cur_rois = acid_rois[acid_rois[:, 0] == i]
        #     cur_rois = cur_rois + acid_offsets_added[i]
        #     acid_iodine_rois.append(cur_rois)
        # acid_iodine_rois = torch.cat(acid_iodine_rois)
        acid_iodine_rois = acid_rois.clone()
        acid_bbox_feats = self.bbox_roi_extractor(acid_feats[:self.bbox_roi_extractor.num_inputs], acid_rois)
        acid_iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], acid_iodine_rois)
        acid_proposal_offsets = self._offset_forward(torch.cat([acid_bbox_feats, acid_iodine_bbox_feats], dim = 1), stage = 1)
        acid_proposal_offsets_scaled = acid_proposal_offsets * torch.cat(
            [acid_rois[:, 3, None] - acid_rois[:, 1, None], acid_rois[:, 4, None] - acid_rois[:, 2, None]], dim = 1)
        acid_proposal_offsets_added = torch.cat(
            [acid_proposal_offsets_scaled.new_zeros(acid_proposal_offsets_scaled.shape[0], 1), acid_proposal_offsets_scaled,
             acid_proposal_offsets_scaled], dim = 1)
        acid_iodine_rois = acid_iodine_rois + acid_proposal_offsets_added
        acid_iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], acid_iodine_rois)
        acid_bbox_feats = acid_bbox_feats + acid_iodine_bbox_feats

        # # iodine
        # iodine_offsets_added = torch.cat(
        #     [global_offsets_scaled.new_zeros((global_offsets_scaled.shape[0], 1)), -global_offsets_scaled, -global_offsets_scaled],
        #     dim = 1)
        # iodine_acid_rois = []
        # for i in range(len(iodine_offsets_added)):
        #     cur_rois = iodine_rois[iodine_rois[:, 0] == i]
        #     cur_rois = cur_rois + iodine_offsets_added[i]
        #     iodine_acid_rois.append(cur_rois)
        # iodine_acid_rois = torch.cat(iodine_acid_rois)
        iodine_acid_rois = acid_rois.clone()
        iodine_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], iodine_rois)
        iodine_acid_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], iodine_acid_rois)
        iodine_proposal_offsets = self._offset_forward(torch.cat([iodine_bbox_feats, iodine_acid_bbox_feats], dim = 1), stage = 1)
        iodine_proposal_offsets_scaled = iodine_proposal_offsets * torch.cat(
            [iodine_rois[:, 3, None] - iodine_rois[:, 1, None], iodine_rois[:, 4, None] - iodine_rois[:, 2, None]], dim = 1)
        iodine_proposal_offsets_added = torch.cat(
            [iodine_proposal_offsets_scaled.new_zeros((iodine_proposal_offsets_scaled.shape[0], 1)), iodine_proposal_offsets_scaled,
             iodine_proposal_offsets_scaled],
            dim = 1)
        iodine_acid_rois = iodine_acid_rois + iodine_proposal_offsets_added
        iodine_acid_bbox_feats = self.bbox_roi_extractor(iodine_feats[:self.bbox_roi_extractor.num_inputs], iodine_acid_rois)
        iodine_bbox_feats = iodine_bbox_feats + iodine_acid_bbox_feats

        if self.with_shared_head:
            acid_bbox_feats = self.shared_head(acid_bbox_feats)
            iodine_bbox_feats = self.shared_head(iodine_bbox_feats)
        acid_cls_score, iodine_cls_score, acid_bbox_pred, iodine_bbox_pred = self.bbox_head(acid_bbox_feats, iodine_bbox_feats)

        bbox_results = dict(acid_cls_score = acid_cls_score, acid_bbox_pred = acid_bbox_pred, acid_bbox_feats = acid_bbox_feats,
                            acid_proposal_offsets = acid_proposal_offsets,
                            iodine_cls_score = iodine_cls_score, iodine_bbox_pred = iodine_bbox_pred, iodine_bbox_feats = iodine_bbox_feats,
                            iodine_proposal_offsets = iodine_proposal_offsets, global_offsets = global_offsets, )
        return bbox_results

    def _offset_forward(self, feat, stage = 0):
        feat = self.offset_pool(feat)
        feat = feat.reshape(feat.shape[0], -1)
        for m in self.offset_modules[stage]:
            feat = m(feat)
        feat = feat.reshape(*feat.shape[:-1], -1, 3)
        return feat

    def _bbox_forward_train(self, acid_feats, iodine_feats, acid_sampling_results, iodine_sampling_results, acid_gt_bboxes,
                            iodine_gt_bboxes, acid_gt_labels, iodine_gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""
        acid_rois = bbox2roi([res.bboxes for res in acid_sampling_results])
        iodine_rois = bbox2roi([res.bboxes for res in iodine_sampling_results])
        bbox_results = self._bbox_forward(acid_feats, iodine_feats, acid_rois, iodine_rois, img_metas)

        bbox_targets = self.bbox_head.get_targets(acid_sampling_results, iodine_sampling_results,
                                                  acid_gt_bboxes, iodine_gt_bboxes,
                                                  acid_gt_labels, iodine_gt_labels,
                                                  img_metas, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['acid_cls_score'], bbox_results['iodine_cls_score'],
                                        bbox_results['acid_bbox_pred'], bbox_results['iodine_bbox_pred'],
                                        bbox_results['acid_proposal_offsets'], bbox_results['iodine_proposal_offsets'],
                                        bbox_results['global_offsets'],
                                        acid_rois, iodine_rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox = loss_bbox)
        return bbox_results

    def simple_test(self,
                    acid_feats, iodine_feats,
                    acid_proposal_list, iodine_proposal_list,
                    img_metas,
                    acid_proposals = None, iodine_proposals = None,
                    rescale = False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        acid_det_bboxes, iodine_det_bboxes, acid_det_labels, iodine_det_labels = \
            self.simple_test_bboxes(acid_feats, iodine_feats, img_metas, acid_proposal_list, iodine_proposal_list, self.test_cfg,
                                    rescale = rescale)

        return ([bbox2result(acid_det_bboxes[i], acid_det_labels[i], self.bbox_head.num_classes) for i in range(len(acid_det_bboxes))],
                [bbox2result(iodine_det_bboxes[i], iodine_det_labels[i], self.bbox_head.num_classes) for i in
                 range(len(iodine_det_bboxes))])

    def simple_test_bboxes(self,
                           acid_feats, iodine_feats,
                           img_metas,
                           acid_proposals, iodine_proposals,
                           rcnn_test_cfg,
                           rescale = False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        acid_rois = bbox2roi(acid_proposals)
        iodine_rois = bbox2roi(iodine_proposals)

        if acid_rois.shape[0] == 0:
            # There is no proposal in the whole batch
            return [acid_rois.new_zeros(0, 5)] * len(acid_proposals), [acid_rois.new_zeros((0,), dtype = torch.long)] * len(
                acid_proposals), [iodine_rois.new_zeros(0, 5)] * len(iodine_proposals), [
                       iodine_rois.new_zeros((0,), dtype = torch.long)] * len(iodine_proposals)

        bbox_results = self._bbox_forward(acid_feats, iodine_feats, acid_rois, iodine_rois, img_metas)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        acid_cls_score = bbox_results['acid_cls_score']
        acid_bbox_pred = bbox_results['acid_bbox_pred']
        acid_num_proposals_per_img = tuple(len(p) for p in acid_proposals)
        acid_rois = acid_rois.split(acid_num_proposals_per_img, 0)
        acid_cls_score = acid_cls_score.split(acid_num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if acid_bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(acid_bbox_pred, torch.Tensor):
                acid_bbox_pred = acid_bbox_pred.split(acid_num_proposals_per_img, 0)
            else:
                acid_bbox_pred = self.bbox_head.bbox_pred_split(
                    acid_bbox_pred, acid_num_proposals_per_img)
        else:
            acid_bbox_pred = (None,) * len(acid_proposals)

        # apply bbox post-processing to each image individually
        acid_det_bboxes = []
        acid_det_labels = []
        for i in range(len(acid_proposals)):
            if acid_rois[i].shape[0] == 0:
                # There is no proposal in the single image
                acid_det_bbox = acid_rois[i].new_zeros(0, 5)
                acid_det_label = acid_rois[i].new_zeros((0,), dtype = torch.long)
            else:
                acid_det_bbox, acid_det_label = self.bbox_head.get_bboxes(
                    acid_rois[i],
                    acid_cls_score[i],
                    acid_bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale = rescale,
                    cfg = rcnn_test_cfg)
            acid_det_bboxes.append(acid_det_bbox)
            acid_det_labels.append(acid_det_label)

        # split batch bbox prediction back to each image
        iodine_cls_score = bbox_results['iodine_cls_score']
        iodine_bbox_pred = bbox_results['iodine_bbox_pred']
        iodine_num_proposals_per_img = tuple(len(p) for p in iodine_proposals)
        iodine_rois = iodine_rois.split(iodine_num_proposals_per_img, 0)
        iodine_cls_score = iodine_cls_score.split(iodine_num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if iodine_bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(iodine_bbox_pred, torch.Tensor):
                iodine_bbox_pred = iodine_bbox_pred.split(iodine_num_proposals_per_img, 0)
            else:
                iodine_bbox_pred = self.bbox_head.bbox_pred_split(
                    iodine_bbox_pred, iodine_num_proposals_per_img)
        else:
            iodine_bbox_pred = (None,) * len(iodine_proposals)

        # apply bbox post-processing to each image individually
        iodine_det_bboxes = []
        iodine_det_labels = []
        for i in range(len(iodine_proposals)):
            if iodine_rois[i].shape[0] == 0:
                # There is no proposal in the single image
                iodine_det_bbox = iodine_rois[i].new_zeros(0, 5)
                iodine_det_label = iodine_rois[i].new_zeros((0,), dtype = torch.long)
            else:
                iodine_det_bbox, iodine_det_label = self.bbox_head.get_bboxes(
                    iodine_rois[i],
                    iodine_cls_score[i],
                    iodine_bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale = rescale,
                    cfg = rcnn_test_cfg)
            iodine_det_bboxes.append(iodine_det_bbox)
            iodine_det_labels.append(iodine_det_label)
        return acid_det_bboxes, iodine_det_bboxes, acid_det_labels, iodine_det_labels

import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin
from mmcv.cnn import normal_init


class PrimAuxAttention(nn.Module):
    #TODO augment feature mutually

    def __init__(self, in_channels, out_channels, num_levels=5, shared=False):
        #TODO add support when in_channels is a list 
        super(PrimAuxAttention, self).__init__()
        assert isinstance(in_channels, int), "type of in_channels must be int"
        assert isinstance(out_channels, int), "type of out_channels must be int"

        self.shared = shared
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_layers()


    def init_layers(self):
        if self.shared:
            self.att = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.att_list = nn.ModuleList([
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
                for _ in range(self.num_levels)
            ]) 


    def init_weights(self):
        if self.shared:
            normal_init(self.att, std=0.01)
        else:
            for i in range(self.num_levels):
                normal_init(self.att_list[i], std=0.01)


    def forward(self, prim_feats, aux_feats):
        """using one feature to augment another feature

        Args:
            prim_feats (list[tensor]): features in different levels 
            aux_feats (list[tensor]): features in different levels

        Returns:
            list[tensor]: features augmented
        """
        if self.shared:
            aug_feats = [
                (self.att(aux_feats[i]).sigmoid() + 1) * prim_feats[i]
                for i in range(self.num_levels)
            ]
        else:
            aug_feats = [
                (self.att_list[i](aux_feats[i]).sigmoid() + 1) * prim_feats[i]
                for i in range(self.num_levels)
            ]
        
        return aug_feats


class PrimSelfAttention(nn.Module):


    def __init__(self, in_channels, out_channels, num_levels=5, shared=False):

        super(PrimSelfAttention, self).__init__()
        assert isinstance(in_channels, int), "type of in_channels must be int"
        assert isinstance(out_channels, int), "type of out_channels must be int"

        self.shared = shared
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_layers()


    def init_layers(self):

        pass


    def init_weights(self):

        pass


    def forward(self, prim_feats, aux_feats):


        pass


class ProposalOffset(nn.Module):


    def __init__(self, in_channels, out_channels, roi_feat_area):
        super(ProposalOffset, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.roi_feat_area = roi_feat_area
        self.init_layers()


    def init_layers(self):
        self.offset = nn.ModuleList([
            nn.Linear(self.roi_feat_area * self.in_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, 2)
        ])


    def init_weights(self):
        for m in self.offset:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


    def forward(self, x):
        x = x.view(-1, self.roi_feat_area * self.in_channels)
        for m in self.offset:
            x = m(x)

        return x


@HEADS.register_module()
class DualCervixPrimAuxRoiHead(BaseRoIHead, BBoxTestMixin):

    def __init__(self, 
                 attention_cfg,
                 offset_cfg,
                 prim_bbox_roi_extractor,
                 aux_bbox_roi_extractor, 
                 bbox_head,
                 train_cfg=None, 
                 test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_attention(attention_cfg)
        self.init_proposalOffset(offset_cfg)
        self.init_bbox_head(prim_bbox_roi_extractor, aux_bbox_roi_extractor, bbox_head)
        self.init_assigner_sampler()

        self.init_weights()


    def init_attention(self, attention_cfg):
        self.attention = PrimAuxAttention(**attention_cfg)


    def init_proposalOffset(self, offset_cfg):
        self.proposalOffset = ProposalOffset(**offset_cfg)


    def init_bbox_head(self, prim_bbox_roi_extractor, aux_bbox_roi_extractor, bbox_head):
        self.prim_bbox_roi_extractor = build_roi_extractor(prim_bbox_roi_extractor)
        self.aux_bbox_roi_extractor = build_roi_extractor(aux_bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)


    def init_mask_head(self):
        raise "not implemented"


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
    

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    
    def init_weights(self):
        self.attention.init_weights()
        self.proposalOffset.init_weights()
        self.prim_bbox_roi_extractor.init_weights()
        self.aux_bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()
        

    def forward_dummy(self):

        pass


    def forward_train(self, 
                      prim_feats, aux_feats,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore):
        """
        Args:
            prim_feats/aux_feats (list[Tensor]): list of multi-level img features.
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

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.attention:
            prim_feats = self.attention(prim_feats, aux_feats)

        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            # print("proposal_list[i].shape", proposal_list[i].shape)
            # print(proposal_list[i][:5])
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in prim_feats])
            sampling_results.append(sampling_result)

        losses = dict()
        bbox_results = self._bbox_forward_train(prim_feats, aux_feats, 
                                                sampling_results,
                                                gt_bboxes, gt_labels)
        losses.update(bbox_results['loss_bbox'])

        return losses


    def _bbox_forward(self, prim_feats, aux_feats, rois):
        """Box head forward function used in both training and testing."""
        prim_bbox_feats = self.prim_bbox_roi_extractor(
            prim_feats[:self.prim_bbox_roi_extractor.num_inputs], rois)

        offset = self.proposalOffset(prim_bbox_feats)
        # offfset reshape [num_img * num_proposal, 2] -> [num_img * num_proposal, 2, out_size, out_size] 
        n = rois.shape[0]
        out_size = prim_bbox_feats.shape[-1]
        offset = offset.view(n, 2, 1, 1).repeat(1, 1, out_size, out_size)
        aux_bbox_feats = self.aux_bbox_roi_extractor(aux_feats, rois, offset)

        bbox_feats = torch.cat([prim_bbox_feats, aux_bbox_feats], dim=1)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)      
        
        return bbox_results


    def _bbox_forward_train(self, prim_feats, aux_feats, sampling_results, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(prim_feats, aux_feats, rois)
        
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results


    def simple_test_bboxes(self, 
                           prim_feats, aux_feats,
                           img_metas, 
                           proposals, 
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        prim_feats = self.attention(prim_feats, aux_feats)
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(prim_feats, aux_feats, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
        

    def simple_test(self, 
                    prim_feats, aux_feats, 
                    proposal_list, 
                    img_metas, 
                    proposals=None, 
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            prim_feats, aux_feats, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]
        #! 每个张图像的不同类别的检测框分开保存的
        return bbox_results


@HEADS.register_module()
class DualCervixDualDetPrimAuxRoiHead(BaseRoIHead, BBoxTestMixin):


    def __init__(self, 
                attention_cfg, 
                offset_cfg,
                prim_bbox_roi_extractor,
                aux_bbox_roi_extractor, 
                bridge_bbox_droi_extractor,
                prim_bbox_head,
                aux_bbox_head,
                train_cfg=None, 
                test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_attention(attention_cfg)
        self.init_proposalOffset(offset_cfg)
        self.init_bbox_head(prim_bbox_roi_extractor, aux_bbox_roi_extractor, bridge_bbox_droi_extractor, prim_bbox_head, aux_bbox_head)
        self.init_assigner_sampler()

        self.init_weights()


    def init_attention(self, attention_cfg):
        self.attention = PrimAuxAttention(**attention_cfg)


    def init_proposalOffset(self, offset_cfg):
        self.proposalOffset = ProposalOffset(**offset_cfg)


    def init_bbox_head(self, 
                       prim_bbox_roi_extractor, aux_bbox_roi_extractor, 
                       bridge_bbox_droi_extractor, 
                       prim_bbox_head, aux_bbox_head):
        self.prim_bbox_roi_extractor = build_roi_extractor(prim_bbox_roi_extractor)
        self.aux_bbox_roi_extractor = build_roi_extractor(aux_bbox_roi_extractor)
        self.bbox_droi_extractor = build_roi_extractor(bridge_bbox_droi_extractor)
        self.prim_bbox_head = build_head(prim_bbox_head)
        self.aux_bbox_head = build_head(aux_bbox_head)


    def init_mask_head(self):
        raise "not implemented"


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)


    def init_weights(self):
        self.attention.init_weights()
        self.proposalOffset.init_weights()
        self.prim_bbox_roi_extractor.init_weights()
        self.aux_bbox_roi_extractor.init_weights()
        self.bridge_bbox_droi_extractor.init_weights()
        self.prim_bbox_head.init_weights()
        self.aux_bbox_head.init_weights()


    def forward_dummy(self):
        raise NotImplementedError


    def forward_train(self, 
                      prim_feats, aux_feats,
                      img_metas,
                      prim_proposal_list, aux_proposal_list,
                      prim_gt_bboxes, prim_gt_labels, prim_gt_bboxes_ignore,
                      aux_gt_bboxes, aux_gt_labels, aux_gt_bboxes_ignore):
        #! attention
        if self.attention:
            prim_feats = self.attention(prim_feats, aux_feats)

        num_imgs = len(img_metas)

        # aux部分给预测框
        if aux_gt_bboxes_ignore is None:
            aux_gt_bboxes_ignore = [None for _ in range(num_imgs)]

        axu_sampling_results = []
        for i in range(num_imgs):
            aux_assign_result = self.bbox_assigner.assign(
                aux_proposal_list[i], aux_gt_bboxes[i], aux_gt_bboxes_ignore[i], aux_gt_labels[i])
            aux_sampling_result = self.bbox_sampler(
                aux_assign_result,
                aux_proposal_list[i],
                aux_gt_bboxes[i],
                aux_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in aux_feats])
            aux_sampling_results.append(aux_sampling_result)
    
        # prim部分
        if prim_gt_bboxes_ignore is None:
            prim_gt_bboxes_ignore = [None for _ in range(num_imgs)]

        prim_sampling_results = []
        for i in range(num_imgs):
            prim_assign_result = self.bbox_assigner.assign(
                prim_proposal_list[i], prim_gt_bboxes[i], prim_gt_bboxes_ignore[i],
                prim_gt_labels[i])
            prim_sampling_result = self.bbox_sampler.sample(
                prim_assign_result,
                prim_proposal_list[i],
                prim_gt_bboxes[i],
                prim_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in prim_feats])
            prim_sampling_results.append(prim_sampling_result)

        losses = dict()
        bbox_results = self._bbox_forward_train(prim_feats, aux_feats, 
                                                prim_sampling_results, aux_sampling_results,
                                                prim_gt_bboxes, aux_gt_bboxes,
                                                prim_gt_labels, aux_gt_labels)
        losses.update(bbox_results['prim_loss_bbox'])
        losses.update(bbox_results['aux_loss_bbox'])

        return losses


    def _bbox_forward(self, prim_feats, aux_feats, prim_rois, aux_rois):
        aux_bbox_feats = self.aux_bbox_roi_extractor(
            aux_feats[:self.aux_bbox_roi_extractor.num_inputs], aux_rois)
        aux_cls_score, aux_bbox_pred = self.aux_bbox_head(aux_bbox_feats)

        prim_bbox_feats = self.prim_bbox_roi_extractor(
            prim_feats[:self.prim_bbox_roi_extractor.num_inputs], prim_rois)
        offset = self.proposalOffset(prim_bbox_feats)
        n = rois.shape[0]

        out_size = prim_bbox_feats.shape[-1]
        offset = offset.view(n, 2, 1, 1).repeat(1, 1, out_size, out_size)
        bridge_bbox_feats = self.bbox_droi_extractor(aux_feats, prim_rois, offset)
        prim_bbox_feats = torch.cat([prim_bbox_feats, bridge_bbox_feats], dim=1)
        prim_cls_score, prim_bbox_pred = self.prim_bbox_head(prim_bbox_feats)

        bbox_results = dict(
            prim_cls_score=prim_cls_score, prim_bbox_pred=prim_bbox_pred, prim_bbox_feats=prim_bbox_feats,
            aux_cls_score=aux_cls_score, aux_bbox_pred=aux_bbox_pred, aux_bbox_feats=aux_bbox_feats)
        
        return bbox_results


    def _bbox_forward_train(self, prim_feats, aux_feats, 
                            prim_sampling_results, aux_sampling_results,
                            prim_gt_bboxes, aux_gt_bboxes,
                            prim_gt_labels, aux_gt_labels):
        prim_rois = bbox2roi([res.bboxes for res in prim_sampling_results])
        aux_rois = bbox2roi([res.bboxes for res in aux_sampling_results])
        
        bbox_results = self._bbox_forward(prim_feats, aux_feats, prim_rois, aux_rois)
        prim_targets = self.prim_bbox_head.get_targets(prim_sampling_results, prim_gt_bboxes, prim_gt_labels, self.train_cfg)
        aux_targets = self.aux_bbox_head.get_targets(aux_sampling_results, aux_gt_bboxes, aux_gt_labels, self.train_cfg)

        prim_loss_bbox = self.prim_bbox_head.loss(bbox_results['prim_cls_score'], 
                                                  bbox_results['prim_bbox_pred'], 
                                                  prim_rois, 
                                                  *prim_targets)
        aux_loss_bbox = self.aux_bbox_head.loss(bbox_results['aux_cls_score'],
                                                bbox_results['aux_bbox_pred'],
                                                aux_rois,
                                                *aux_targets)
        bbox_results.update(prim_loss_bbox=prim_loss_bbox)
        bbox_results.update(aux_loss_bbox=aux_loss_bbox)

        return bbox_results
    

    def simple_test_bboxes(self, 
                           prim_feats, aux_feats, 
                           img_metas, 
                           prim_proposals, aux_proposals, 
                           rcnn_test_cfg, 
                           rescale=False):
        #! attention
        if self.attention:
            prim_feats = self.attention(prim_feats, aux_feats)

        prim_rois = bbox2roi([res.bboxes for res in prim_sampling_results])
        aux_rois = bbox2roi([res.bboxes for res in aux_sampling_results])
        bbox_results = self._bbox_forward(prim_feats, aux_feats, prim_rois, aux_rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # aux
        aux_cls_score = bbox_results['aux_cls_score']
        aux_bbox_pred = bbox_results['aux_bbox_pred']
        aux_num_proposals_per_img = tuple(len(p) for p in aux_proposals)
        aux_rois = aux_rois.split(aux_num_proposals_per_img, 0)
        aux_cls_score = aux_cls_score.split(aux_num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if aux_bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(aux_bbox_pred, torch.Tensor):
                aux_bbox_pred = aux_bbox_pred.split(aux_num_proposals_per_img, 0)
            else:
                aux_bbox_pred = self.aux_bbox_head.bbox_pred_split(
                    aux_bbox_pred, aux_num_proposals_per_img)
        else:
            aux_bbox_pred = (None, ) * len(aux_proposals)

        # apply bbox post-processing to each image individually
        aux_det_bboxes = []
        aux_det_labels = []
        for i in range(len(aux_proposals)):
            aux_det_bbox, aux_det_label = self.aux_bbox_head.get_bboxes(
                aux_rois[i],
                aux_cls_score[i],
                aux_bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            aux_det_bboxes.append(aux_det_bbox)
            aux_det_labels.append(aux_det_label)        

        # prim
        prim_cls_score = bbox_results['prim_cls_score']
        prim_bbox_pred = bbox_results['prim_bbox_pred']
        prim_num_proposals_per_img = tuple(len(p) for p in prim_proposals)
        prim_rois = prim_rois.split(prim_num_proposals_per_img, 0)
        prim_cls_score = prim_cls_score.split(prim_num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if prim_bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(prim_bbox_pred, torch.Tensor):
                prim_bbox_pred = prim_bbox_pred.split(prim_num_proposals_per_img, 0)
            else:
                prim_bbox_pred = self.prim_bbox_head.bbox_pred_split(
                    prim_bbox_pred, prim_num_proposals_per_img)
        else:
            prim_bbox_pred = (None, ) * len(prim_proposals)

        # apply bbox post-processing to each image individually
        prim_det_bboxes = []
        prim_det_labels = []
        for i in range(len(prim_proposals)):
            prim_det_bbox, prim_det_label = self.prim_bbox_head.get_bboxes(
                prim_rois[i],
                prim_cls_score[i],
                prim_bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            prim_det_bboxes.append(prim_det_bbox)
            prim_det_labels.append(prim_det_label)

        return prim_det_bboxes, prim_det_labels, aux_det_bboxes, aux_det_labels


    def simple_test(self, 
                    prim_feats, aux_feats, 
                    prim_proposal_list, aux_proposal_list, 
                    img_metas, 
                    prim_proposals=None, aux_proposals=None, 
                    rescale=False):
        (prim_det_bboxes, prim_det_labels, 
        aux_det_bboxes, aux_det_labels) = self.simple_test_bboxes(prim_feats, aux_feats, 
                                                                  img_metas, 
                                                                  prim_proposal_list, aux_proposal_list,
                                                                  self.test_cfg,
                                                                  rescale=rescale)
        prim_results = [
            bbox2result(prim_det_bboxes[i], prim_det_labels[i], 
                        self.prim_bbox_head.num_classes)
            for i in range(len(prim_det_bboxes))
        ]

        aux_results = [
            bbox2result(aux_det_bboxes[i], aux_det_labels[i], 
                        self.aux_bbox_head.num_classes)
            for i in range(len(aux_det_bboxes))
        ]
        #! list中是每张图像的检测框
        return prim_results, aux_results
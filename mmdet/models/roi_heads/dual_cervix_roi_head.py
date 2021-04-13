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
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
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
                           prims_feats, aux_feats,
                           img_metas, 
                           proposals, 
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        prim_feats = self.attention(prim_feats, aux_feats)
        rois = bbox2roi(proposal_list)
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

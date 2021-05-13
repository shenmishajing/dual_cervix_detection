from mmcv.runner.fp16_utils import force_fp32
import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, roi2bbox
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin
from mmcv.cnn import normal_init
from .dual_cervix_roi_head_utils import build_proposaloffset, PrimAuxAttention, build_fpnfeaturefuser
import os
import numpy as np 


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
        self.proposalOffset = build_proposaloffset(**offset_cfg)


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
        aux_bbox_feats = self.aux_bbox_roi_extractor(aux_feats[:self.aux_bbox_roi_extractor.num_inputs], rois, offset)

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
                prim_bbox_roi_extractor,
                aux_bbox_roi_extractor, 
                prim_bbox_head,
                aux_bbox_head,
                bridge_bbox_roi_extractor=None,
                attention_cfg=None, 
                offset_cfg=None,
                fpn_fuser_cfg=None,
                train_cfg=None, 
                test_cfg=None):
        super(BaseRoIHead, self).__init__()

        if (offset_cfg is None and bridge_bbox_roi_extractor is not None) or (offset_cfg is not None and bridge_bbox_roi_extractor is None):
            raise "offset_cfg and bridge_bbox_roi_extractor must be None or not None at the same time"
        
        if bridge_bbox_roi_extractor is not None:
            self.bridge_bbox_roi_extractor_type = bridge_bbox_roi_extractor["type"]

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_attention(attention_cfg)
        self.init_proposalOffset(offset_cfg)
        self.init_FPNFeatureFuser(fpn_fuser_cfg)
        self.init_bbox_head(prim_bbox_roi_extractor, aux_bbox_roi_extractor, bridge_bbox_roi_extractor, prim_bbox_head, aux_bbox_head)
        self.init_assigner_sampler()

        self.init_weights()
       
        # 用于proposal可视化
        self.proposal_idx = 0


    def init_attention(self, attention_cfg):
        if attention_cfg is not None:
            self.attention = PrimAuxAttention(**attention_cfg)
        else:
            self.attention = None


    def init_proposalOffset(self, offset_cfg):
        if offset_cfg is not None:
            self.proposalOffset = build_proposaloffset(offset_cfg)
        else:
            self.proposalOffset = None


    def init_FPNFeatureFuser(self, fpn_fuser_cfg):
        if fpn_fuser_cfg is not None:
            self.fpn_fuser = build_fpnfeaturefuser(fpn_fuser_cfg)
        else:
            self.fpn_fuser = None


    def init_bbox_head(self, 
                       prim_bbox_roi_extractor, aux_bbox_roi_extractor, 
                       bridge_bbox_roi_extractor, 
                       prim_bbox_head, aux_bbox_head):
        self.prim_bbox_roi_extractor = build_roi_extractor(prim_bbox_roi_extractor)
        self.aux_bbox_roi_extractor = build_roi_extractor(aux_bbox_roi_extractor)
        
        if bridge_bbox_roi_extractor:
            self.bridge_bbox_roi_extractor = build_roi_extractor(bridge_bbox_roi_extractor)
        else:
            self.bridge_bbox_roi_extractor = None
        
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
        if self.attention:
            self.attention.init_weights()
        if self.proposalOffset:
            self.proposalOffset.init_weights()
        if self.fpn_fuser:
            self.fpn_fuser.init_weights()

        self.prim_bbox_roi_extractor.init_weights()
        self.aux_bbox_roi_extractor.init_weights()
        
        if self.bridge_bbox_roi_extractor:
            self.bridge_bbox_roi_extractor.init_weights()
        
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

        aux_sampling_results = []
        for i in range(num_imgs):
            aux_assign_result = self.bbox_assigner.assign(
                aux_proposal_list[i], aux_gt_bboxes[i], aux_gt_bboxes_ignore[i], aux_gt_labels[i])
            aux_sampling_result = self.bbox_sampler.sample(
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
                prim_proposal_list[i], prim_gt_bboxes[i], prim_gt_bboxes_ignore[i], prim_gt_labels[i])
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
        

        if self.bridge_bbox_roi_extractor:
            if self.fpn_fuser:
                prim_bbox_feats_ = self.fpn_fuser(prim_bbox_feats, prim_feats[:self.prim_bbox_roi_extractor.num_inputs], 
                                                aux_feats[:self.aux_bbox_roi_extractor.num_inputs], prim_rois)
                offset = self.proposalOffset(prim_bbox_feats_)
            else:
                offset = self.proposalOffset(prim_bbox_feats)
            
            VIS = False
            if VIS:
                # print(prim_rois[0, :])
                # print(prim_rois.shape)
                # print(offset[:5, :, 0, 0])
                # print(offset.shape)
                self.proposals_vis(prim_rois, offset)

            bridge_bbox_feats = self._extract_bridge_bbox_feats(prim_bbox_feats, aux_feats, prim_rois, offset)

            prim_bbox_feats = torch.cat([prim_bbox_feats, bridge_bbox_feats], dim=1)

        prim_cls_score, prim_bbox_pred = self.prim_bbox_head(prim_bbox_feats)

        bbox_results = dict(
            prim_cls_score=prim_cls_score, prim_bbox_pred=prim_bbox_pred, prim_bbox_feats=prim_bbox_feats,
            aux_cls_score=aux_cls_score, aux_bbox_pred=aux_bbox_pred, aux_bbox_feats=aux_bbox_feats)
        
        return bbox_results


    def _extract_bridge_bbox_feats(self, prim_bbox_feats, aux_feats, prim_rois, offset):
        if self.bridge_bbox_roi_extractor_type == "SingleDeformRoIExtractor":
            n = prim_rois.shape[0]
            out_size = prim_bbox_feats.shape[-1]
            offset = offset.view(n, 2, 1, 1).repeat(1, 1, out_size, out_size)
            bridge_bbox_feats = self.bridge_bbox_roi_extractor(aux_feats[:self.bridge_bbox_roi_extractor.num_inputs], prim_rois, offset)

        elif self.bridge_bbox_roi_extractor_type == "SingleRoIExtractor":
            prim_rois = self.proposalOffset.apply_offset(prim_rois, offsets)
            bridge_bbox_feats = self.bridge_bbox_roi_extractor(aux_feats[:self.bridge_bbox_roi_extractor.num_inputs], prim_rois)

        else:
            raise "roi_extractor_type = {} is not support".format(self.bridge_bbox_roi_extractor_type)
        
        return bridge_bbox_feats


    def _bbox_forward_train(self, prim_feats, aux_feats, 
                            prim_sampling_results, aux_sampling_results,
                            prim_gt_bboxes, aux_gt_bboxes,
                            prim_gt_labels, aux_gt_labels):
        prim_rois = bbox2roi([res.bboxes for res in prim_sampling_results])
        aux_rois = bbox2roi([res.bboxes for res in aux_sampling_results])
        print("prim_rois", prim_rois)
        exit(-1)
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
        prim_loss_bbox = {"prim_" + k: v for k,v in prim_loss_bbox.items()}
        aux_loss_bbox = {"aux_" + k: v for k,v in aux_loss_bbox.items()}
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

        prim_rois = bbox2roi(prim_proposals)
        aux_rois = bbox2roi(aux_proposals)
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


    def proposals_vis(self, rois, offset):
        print(rois.shape, offset.shape)

        gamma = 0.1
        offset_tmp = gamma * (rois[:, 3:5] - rois[:, 1:3]) * offset
        aux_proposal_coord = torch.cat([
            rois[:, 1:3] + offset_tmp, rois[:, 3:5] + offset_tmp
        ], dim=-1)
        
        prim_proposal_coord = rois[:, 1:5]
        exp_dir = "/data2/luochunhua/od/mmdetection/work_dirs/dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_gamma3_acid_hsil/proposals/"
       
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        aux_proposal_path = os.path.join(exp_dir, "aux_{}.npz".format(str(self.proposal_idx)))
        prim_proposal_path = os.path.join(exp_dir, "prim_{}.npz".format(str(self.proposal_idx)))
        np.savez(aux_proposal_path, aux_proposal_coord.cpu().numpy())
        np.savez(prim_proposal_path, prim_proposal_coord.cpu().numpy())

        self.proposal_idx += 1


@HEADS.register_module()
class DualCervixDualDetPrimAuxAuxLossRoiHead(DualCervixDualDetPrimAuxRoiHead):

    def __init__(self, 
                 prim_bbox_roi_extractor, aux_bbox_roi_extractor, 
                 prim_bbox_head, aux_bbox_head, 
                 bridge_bbox_roi_extractor, 
                 attention_cfg, 
                 offset_cfg, 
                 fpn_fuser_cfg,
                 aux_loss_cfg, 
                 train_cfg, test_cfg):

        self.offset_type = offset_cfg.get("type")
        super().__init__(prim_bbox_roi_extractor, 
                         aux_bbox_roi_extractor, 
                         prim_bbox_head, 
                         aux_bbox_head, 
                         bridge_bbox_roi_extractor=bridge_bbox_roi_extractor, 
                         attention_cfg=attention_cfg, 
                         offset_cfg=offset_cfg, 
                         fpn_fuser_cfg=fpn_fuser_cfg, 
                         train_cfg=train_cfg, test_cfg=test_cfg)

        self.aux_loss = build_loss(aux_loss_cfg)
        

    def forward_train(self,
                      prim_feats, aux_feats,
                      img_metas,
                      prim_proposal_list, aux_proposal_list,
                      prim_gt_bboxes, prim_gt_labels, prim_gt_bboxes_ignore,
                      aux_gt_bboxes, aux_gt_labels, aux_gt_bboxes_ignore):

        assert aux_proposal_list == None

        #! attention
        if self.attention:
            prim_feats = self.attention(prim_feats, aux_feats)

        num_imgs = len(img_metas)
    
        # prim部分
        if prim_gt_bboxes_ignore is None:
            prim_gt_bboxes_ignore = [None for _ in range(num_imgs)]

        prim_sampling_results = []
        for i in range(num_imgs):
            prim_assign_result = self.bbox_assigner.assign(
                prim_proposal_list[i], prim_gt_bboxes[i], prim_gt_bboxes_ignore[i], prim_gt_labels[i])
            prim_sampling_result = self.bbox_sampler.sample(
                prim_assign_result,
                prim_proposal_list[i],
                prim_gt_bboxes[i],
                prim_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in prim_feats])
            prim_sampling_results.append(prim_sampling_result)

        losses = dict()
        bbox_results = self._bbox_forward_train(prim_feats, aux_feats, 
                                                prim_sampling_results, None,
                                                prim_gt_bboxes, aux_gt_bboxes,
                                                prim_gt_labels, aux_gt_labels,
                                                aux_gt_bboxes_ignore)
        losses.update(bbox_results['prim_loss_bbox'])
        losses.update(bbox_results['aux_loss_bbox'])
        losses.update(bbox_results['aux_loss'])

        return losses

    
    def _bbox_forward_train(self, 
                            prim_feats, aux_feats,
                            prim_sampling_results, aux_sampling_results, 
                            prim_gt_bboxes, aux_gt_bboxes, 
                            prim_gt_labels, aux_gt_labels, 
                            aux_gt_bboxes_ignore):
        prim_rois = bbox2roi([res.bboxes for res in prim_sampling_results])

        bbox_results, aux_rois = self._bbox_forward(prim_feats, aux_feats, prim_rois)
        prim_targets = self.prim_bbox_head.get_targets(prim_sampling_results, prim_gt_bboxes, prim_gt_labels, self.train_cfg)

        #! aux_sampling_results
        aux_proposal_list = roi2bbox(aux_rois)
        aux_sampling_results = []
        for i in range(aux_proposal_list):
            aux_assign_result = self.bbox_assigner.assign(
                aux_proposal_list[i], aux_gt_bboxes[i], aux_gt_bboxes_ignore[i], aux_gt_labels[i])
            aux_sampling_result = self.bbox_sampler.sample(
                aux_assign_result,
                aux_proposal_list[i],
                aux_gt_bboxes[i],
                aux_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in aux_feats])
            aux_sampling_results.append(aux_sampling_result)
        aux_targets = self.aux_bbox_head.get_targets(aux_sampling_results, aux_gt_bboxes, aux_gt_labels, self.train_cfg)

        aux_rois_box = aux_rois[:, 1:]
        aux_loss = self._aux_loss(aux_rois_box, *aux_targets, self.prim_bbox_head.num_classes)

        prim_loss_bbox = self.prim_bbox_head.loss(bbox_results['prim_cls_score'], 
                                                  bbox_results['prim_bbox_pred'], 
                                                  prim_rois, 
                                                  *prim_targets)
        aux_loss_bbox = self.aux_bbox_head.loss(bbox_results['aux_cls_score'],
                                                bbox_results['aux_bbox_pred'],
                                                aux_rois,
                                                *aux_targets)
        prim_loss_bbox = {"prim_" + k: v for k,v in prim_loss_bbox.items()}
        aux_loss_bbox = {"aux_" + k: v for k,v in aux_loss_bbox.items()}
        bbox_results.update(prim_loss_bbox=prim_loss_bbox)
        bbox_results.update(aux_loss_bbox=aux_loss_bbox)
        bbox_results.update(aux_loss=aux_loss)

        return bbox_results


    @force_fp32(apply_to=('bbox_pred', ))
    def _aux_loss(self, 
                  bbox_pred,
                  labels,
                  labels_weights, 
                  bbox_targets, 
                  bbox_weights, 
                  num_classes,
                  reduction_override=None):
        losses = dict()
        bg_class_ind = num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)

        if pos_inds.any():
            pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1, 4)[pos_inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]

        losses['loss_bbox'] = self.aux_loss(
            pos_bbox_pred,
            bbox_targets[pos_inds.type(torch.bool)],
            bbox_weights[pos_inds.type(torch.bool)],
            avg_factor=bbox_targets.size(0),
            reduction_override=reduction_override)

        return losses


    def _bbox_forward(self, prim_feats, aux_feats, prim_rois):
        prim_bbox_feats = self.prim_bbox_roi_extractor(
            prim_feats[:self.prim_bbox_roi_extractor.num_inputs], prim_rois)
        
        if self.fpn_fuser:
            prim_bbox_feats_ = self.fpn_fuser(prim_bbox_feats, prim_feats[:self.prim_bbox_roi_extractor.num_inputs], 
                                            aux_feats[:self.aux_bbox_roi_extractor.num_inputs], prim_rois)
            offset = self.proposalOffset(prim_bbox_feats_)
        else:
            offset = self.proposalOffset(prim_bbox_feats)
        
        bridge_bbox_feats, aux_rois = self._extract_bridge_bbox_feats(prim_bbox_feats, aux_feats, prim_rois, offset)

        aux_bbox_feats = self.aux_bbox_roi_extractor(
            aux_feats[:self.aux_bbox_roi_extractor.num_inputs], aux_rois)
        aux_cls_score, aux_bbox_pred = self.aux_bbox_head(aux_bbox_feats)

        prim_bbox_feats = torch.cat([prim_bbox_feats, bridge_bbox_feats], dim=1)        
        prim_cls_score, prim_bbox_pred = self.prim_bbox_head(prim_bbox_feats)

        bbox_results = dict(
            prim_cls_score=prim_cls_score, prim_bbox_pred=prim_bbox_pred, prim_bbox_feats=prim_bbox_feats,
            aux_cls_score=aux_cls_score, aux_bbox_pred=aux_bbox_pred, aux_bbox_feats=aux_bbox_feats)
        
        return bbox_results, aux_rois


    def _extract_bridge_bbox_feats(self, prim_bbox_feats, aux_feats, prim_rois, offset):
        if self.bridge_bbox_roi_extractor_type == "SingleDeformRoIExtractor":
            assert self.offset_type == "ProposalOffsetXY"
            aux_rois_coord = self.proposalOffset.apply_offset(prim_rois[:, 1:], offset)
            aux_rois = torch.cat([prim_rois[:, 0:1], aux_rois_coord], dim=-1)

            n = prim_rois.shape[0]
            out_size = prim_bbox_feats.shape[-1]
            offset = offset.view(n, 2, 1, 1).repeat(1, 1, out_size, out_size)
            bridge_bbox_feats = self.bridge_bbox_roi_extractor(aux_feats[:self.bridge_bbox_roi_extractor.num_inputs], prim_rois, offset)

        elif self.bridge_bbox_roi_extractor_type == "SingleRoIExtractor":
            aux_rois_coord = self.proposalOffset.apply_offset(prim_rois[:, 1:], offset)
            aux_rois = torch.cat([prim_rois[:, 0:1], aux_rois_coord], dim=-1)
            bridge_bbox_feats = self.bridge_bbox_roi_extractor(aux_feats[:self.bridge_bbox_roi_extractor.num_inputs], aux_rois)

        else:
            raise "roi_extractor_type = {} is not support".format(self.bridge_bbox_roi_extractor_type)
        
        return bridge_bbox_feats, aux_rois


    def simple_test_bboxes(self, 
                           prim_feats, aux_feats,
                           img_metas, 
                           prim_proposals, aux_proposals, 
                           rcnn_test_cfg, 
                           rescale=False):
        # 没有辅助模态没有 rpn
        assert aux_proposals == None
       
        #! attention
        if self.attention:
            prim_feats = self.attention(prim_feats, aux_feats)
        
        prim_rois = bbox2roi(prim_proposals)
        bbox_results, aux_rois = self._bbox_forward(prim_feats, aux_feats, prim_rois)


        # 预测结果的处理
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
        assert aux_proposal_list == None

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
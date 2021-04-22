import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector
from mmcv.runner import auto_fp16


@DETECTORS.register_module()
class FasterPrimAuxDetector(TwoStageDetector):


    def __init__(self, 
                 prim_backbone, aux_backbone,
                 prim_neck, aux_neck,
                 rpn_head,
                 roi_head,
                 aug_acid=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.prim_backbone = build_backbone(prim_backbone)
        self.aux_backbone = build_backbone(aux_backbone)

        self.prim_neck = build_neck(prim_neck)
        self.aux_neck = build_neck(aux_neck)

        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_ = rpn_head.copy()
        rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.rpn_head = build_head(rpn_head_)

        # update train and test cfg here for now
        # TODO: refactor assigner & sampler
        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        self.roi_head = build_head(roi_head)

        self.aug_acid = aug_acid
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        self.prim_backbone.init_weights(pretrained)
        self.aux_backbone.init_weights(pretrained)        
        self.prim_neck.init_weights()
        self.aux_neck.init_weights()
        
        self.rpn_head.init_weights()
        self.roi_head.init_weights()


    def extract_feat(self, prim_img, aux_img):
        prim_feats = self.prim_backbone(prim_img)
        prim_feats = self.prim_neck(prim_feats)

        aux_feats = self.aux_backbone(aux_img)
        aux_feats = self.aux_neck(aux_feats)

        return prim_feats, aux_feats


    def forward_dummy(self, acid_img, iodine_img):
        raise "not impelemented"


    def forward_train(self, 
                      acid_img, iodine_img,
                      img_metas, 
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore=None, iodine_gt_bboxes_ignore=None,
                      acid_proposals=None, iodine_proposals=None,
                      **kwargs):
        
        if self.aug_acid:
            prim_img = acid_img
            aux_img = iodine_img
            gt_bboxes = acid_gt_bboxes
            gt_labels = acid_gt_labels
            gt_bboxes_ignore = acid_gt_bboxes_ignore
            proposal = acid_proposals
        else:
            prim_img = iodine_img
            aux_img = acid_img
            gt_bboxes = iodine_gt_bboxes
            gt_labels = iodine_gt_labels
            gt_bboxes_ignore = iodine_gt_bboxes_ignore
            proposal = iodine_proposals

        prim_feats, aux_feats = self.extract_feat(prim_img, aux_img)

        losses = dict()
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            prim_feats, 
            img_metas, 
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(prim_feats, aux_feats, 
                                                 img_metas, 
                                                 proposal_list, 
                                                 gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(roi_losses)

        return losses


    def forward_test(self, acid_imgs, iodine_imgs, img_metas, **kwargs):
        for var, name in [(acid_imgs, 'acid_imgs'), (iodine_imgs, 'iodine_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')        
        
        num_augs = len(acid_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(acid_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        if self.aug_acid:
            prim_imgs = acid_imgs
            aux_imgs = iodine_imgs
        else:
            prim_imgs = iodine_imgs
            aux_imgs = acid_imgs

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(prim_imgs[0], aux_imgs[0], img_metas[0], **kwargs)
        else:
            raise "aug_test is not support"


    @auto_fp16(apply_to=('acid_img', 'iodine_img'))
    def forward(self, acid_img, iodine_img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(acid_img, iodine_img, img_metas, **kwargs)
        else:
            return self.forward_test(acid_img, iodine_img, img_metas, **kwargs)


    def simple_test(self, prim_img, aux_img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        prim_feats, aug_feats = self.extract_feat(prim_img, aux_img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(prim_feats, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(prim_feats, aug_feats, 
                proposal_list, img_metas, proposals, rescale=rescale)



@DETECTORS.register_module()
class FasterPrimAuxDualDetector(FasterPrimAuxDetector):

    def __init__(self,  
                 prim_backbone, aux_backbone,
                 prim_neck, aux_neck,
                 prim_rpn_head, aux_rpn_head,
                 roi_head,
                 aug_acid=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.dual_det = True
        self.prim_backbone = build_backbone(prim_backbone)
        self.aux_backbone = build_backbone(aux_backbone)

        self.prim_neck = build_neck(prim_neck)
        self.aux_neck = build_neck(aux_neck)

        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None

        prim_rpn_head_ = prim_rpn_head.copy()
        prim_rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.prim_rpn_head = build_head(prim_rpn_head_)
        
        aux_rpn_head_ = aux_rpn_head.copy()
        aux_rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.aux_rpn_head = build_head(aux_rpn_head_)

        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None 
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        self.roi_head = build_head(roi_head)

        self.aug_acid = aug_acid
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        self.prim_backbone.init_weights(pretrained)
        self.prim_neck.init_weights()
        self.prim_rpn_head.init_weights()

        self.aux_backbone.init_weights(pretrained)        
        self.aux_neck.init_weights()
        self.aux_rpn_head.init_weights()

        self.roi_head.init_weights()


    def forward_train(self, 
                      acid_img, iodine_img,
                      img_metas, 
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore=None, iodine_gt_bboxes_ignore=None,
                      acid_proposals=None, iodine_proposals=None,
                      **kwargs):

        if self.aug_acid:
            prim_img = acid_img
            aux_img = iodine_img
            prim_gt_bboxes = acid_gt_bboxes
            prim_gt_labels = acid_gt_labels
            prim_gt_bboxes_ignore = acid_gt_bboxes_ignore
            prim_proposal = acid_proposals

            aux_gt_bboxes = iodine_gt_bboxes
            aux_gt_labels = iodine_gt_labels
            aux_gt_bboxes_ignore = iodine_gt_bboxes_ignore
            aux_proposals = iodine_proposals
        else:
            prim_img = iodine_img
            aux_img = acid_img
            prim_gt_bboxes = iodine_gt_bboxes
            prim_gt_labels = iodine_gt_labels
            prim_gt_bboxes_ignore = iodine_gt_bboxes_ignore
            prim_proposal = iodine_proposals

            aux_gt_bboxes = acid_gt_bboxes
            aux_gt_labels = acid_gt_labels
            aux_gt_bboxes_ignore = acid_gt_bboxes_ignore
            aux_proposals = acid_proposals

        prim_feats, aux_feats = self.extract_feat(prim_img, aux_img)


        losses = dict()
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        prim_rpn_losses, prim_proposal_list = self.prim_rpn_head.forward_train(
            prim_feats, 
            img_metas, 
            prim_gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=prim_gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        # print("prim_rpn_losses", prim_rpn_losses)
        losses.update({"prim_" + k: v  for k,v in prim_rpn_losses.items()})

        aux_rpn_losses, aux_proposal_list = self.aux_rpn_head.forward_train(
            aux_feats, 
            img_metas, 
            aux_gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=aux_gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        # print("aux_rpn_losses", aux_rpn_losses)
        losses.update({"aux_" + k : v for k, v in aux_rpn_losses.items()})

        roi_losses = self.roi_head.forward_train(prim_feats, aux_feats,
                                                 img_metas, 
                                                 prim_proposal_list, aux_proposal_list,
                                                 prim_gt_bboxes, prim_gt_labels, prim_gt_bboxes_ignore,
                                                 aux_gt_bboxes, aux_gt_labels, aux_gt_bboxes_ignore)
        
        losses.update(roi_losses)
        
        return losses

    
    def forward_test(self, acid_imgs, iodine_imgs, img_metas, **kwargs):
        for var, name in [(acid_imgs, 'acid_imgs'), (iodine_imgs, 'iodine_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')        
        
        num_augs = len(acid_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(acid_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        if self.aug_acid:
            prim_imgs = acid_imgs
            aux_imgs = iodine_imgs
        else:
            prim_imgs = iodine_imgs
            aux_imgs = acid_imgs

        if num_augs == 1:
            if 'acid_proposals' in kwargs:
                if self.aug_acid:
                    kwargs['prim_proposals'] = kwargs['acid_proposals'][0]
                    kwargs['aux_proposals'] = kwargs['iodine_proposals'][0]
                else:
                    kwargs['prim_proposals'] = kwargs['iodine_proposals'][0]
                    kwargs['aux_proposals'] = kwargs['acid_proposals'][0]

            return self.simple_test(prim_imgs[0], aux_imgs[0], img_metas[0], **kwargs)
        else:
            raise "aug_test is not support"


    def simple_test(self, 
                    prim_img, aux_img, 
                    img_metas, 
                    prim_proposals=None, aux_proposals=None,
                    rescale=False):
        prim_feats, aux_feats = self.extract_feat(prim_img, aux_img)
        if prim_proposals is None:
            prim_proposal_list = self.prim_rpn_head.simple_test_rpn(prim_feats, img_metas)
            aux_proposal_list = self.aux_rpn_head.simple_test_rpn(aux_feats, img_metas)
        else:
            prim_proposal_list = prim_proposals
            aux_proposal_list = aux_proposals

        return self.roi_head.simple_test(prim_feats, aux_feats, 
                                         prim_proposal_list, aux_proposal_list, 
                                         img_metas,
                                         prim_proposals, aux_proposals,
                                         rescale=rescale)
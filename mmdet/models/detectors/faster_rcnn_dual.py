import warnings

from torch import nn
from mmdet.core import bbox2result
from mmcv.runner import auto_fp16
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNDual(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head_acid=None,roi_head_iodine=None,
                 prim=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)

        if prim is None:
            self.prim = ['acid', 'iodine']
        elif not isinstance(prim, list):
            self.prim = [prim]
        else:
            self.prim = prim

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:

            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)


        if roi_head_acid is not None and roi_head_iodine is not None:

            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head_acid.update(train_cfg=rcnn_train_cfg)
            roi_head_acid.update(test_cfg=test_cfg.rcnn)
            roi_head_acid.pretrained = pretrained
            self.roi_head_acid = build_head(roi_head_acid)

            roi_head_iodine.update(train_cfg=rcnn_train_cfg)
            roi_head_iodine.update(test_cfg=test_cfg.rcnn)
            roi_head_iodine.pretrained = pretrained
            self.roi_head_iodine = build_head(roi_head_iodine)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg



    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        return super(FasterRCNNDual, self).extract_feat(acid_img), super(FasterRCNNDual, self).extract_feat(iodine_img)

    def forward_dummy(self, acid_img, iodine_img):
        raise NotImplementedError

    def forward_train(self,
                      acid_img, iodine_img,
                      img_metas,
                      acid_gt_bboxes, iodine_gt_bboxes,
                      acid_gt_labels, iodine_gt_labels,
                      acid_gt_bboxes_ignore = None, iodine_gt_bboxes_ignore = None,
                      acid_proposals = None, iodine_proposals = None,
                      **kwargs):

        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)

        losses = dict()

        # RPN forward and loss
        if self.rpn_head:

            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            acid_rpn_losses, acid_proposal_list = self.rpn_head.forward_train(
                acid_feats, img_metas, acid_gt_bboxes, gt_labels = None, gt_bboxes_ignore = acid_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'acid' in self.prim:
                for k, v in acid_rpn_losses.items():
                    losses['acid_' + k] = v
            iodine_rpn_losses, iodine_proposal_list = self.rpn_head.forward_train(
                iodine_feats, img_metas, iodine_gt_bboxes, gt_labels = None, gt_bboxes_ignore = iodine_gt_bboxes_ignore,
                proposal_cfg = proposal_cfg)
            if 'iodine' in self.prim:
                for k, v in iodine_rpn_losses.items():
                    losses['iodine_' + k] = v
        else:
            acid_proposal_list = acid_proposals
            iodine_proposal_list = iodine_proposals


        roi_acid_losses = self.roi_head_acid.forward_train(acid_feats = acid_feats, iodine_feats=iodine_feats, img_metas =img_metas, acid_proposal_list =acid_proposal_list, acid_gt_bboxes =acid_gt_bboxes,
                                                           iodine_gt_bboxes=iodine_gt_bboxes, acid_gt_labels =acid_gt_labels,iodine_gt_labels =iodine_gt_labels, acid_gt_bboxes_ignore = None,
                                                           iodine_gt_bboxes_ignore = None, **kwargs)
        roi_iodine_losses = self.roi_head_iodine.forward_train(acid_feats = iodine_feats, iodine_feats=acid_feats, img_metas =img_metas, acid_proposal_list =iodine_proposal_list, acid_gt_bboxes =iodine_gt_bboxes,
                                                               iodine_gt_bboxes=acid_gt_bboxes, acid_gt_labels =iodine_gt_labels,iodine_gt_labels =acid_gt_labels, acid_gt_bboxes_ignore=None,
                                                               iodine_gt_bboxes_ignore=None, **kwargs)
        for k, v in roi_acid_losses.items():
            if any([name in k for name in self.prim]):
                losses[k] = v
        for k, v in roi_iodine_losses.items():
            k=k.replace('acid','iodine')
            if any([name in k for name in self.prim]):
                losses[k] = v

        return losses

    def forward_test(self, acid_imgs, iodine_imgs, img_metas, **kwargs):
        for var, name in [(acid_imgs, 'acid_imgs'), (iodine_imgs, 'iodine_imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(acid_imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(acid_imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(acid_imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(acid_imgs[0], iodine_imgs[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    @auto_fp16(apply_to = ('acid_img', 'iodine_img'))
    def forward(self, acid_img, iodine_img, img_metas, return_loss = True, **kwargs):
        if return_loss:
            return self.forward_train(acid_img, iodine_img, img_metas, **kwargs)
        else:
            return self.forward_test(acid_img, iodine_img, img_metas, **kwargs)

    def simple_test(self, acid_img, iodine_img, img_metas, acid_proposals = None, iodine_proposals = None, rescale = False):
        """Test without augmentation."""
        assert self.roi_head_acid, 'Bbox head must be implemented.'
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)






        # # for visualization feature map
        # import torch
        # def featuremap_2_heatmap(feature_map):
        #     assert isinstance(feature_map, torch.Tensor)
        #     feature_map = feature_map.detach()
        #     heatmap = feature_map[:, 0, :, :] * 0
        #     for c in range(feature_map.shape[1]):
        #         heatmap += feature_map[:, c, :, :]
        #     heatmap = heatmap.cpu().numpy()
        #     heatmap = np.mean(heatmap, axis=0)
        #
        #     heatmap = np.maximum(heatmap, 0)
        #     heatmap /= np.max(heatmap)
        #
        #     return heatmap
        #
        #
        # if img_metas[0]['filename'][0][:-6] == '02895805_2019-07-12':
        #     import cv2
        #     import numpy as np
        #     import os
        #     i = 0
        #     for featuremap in acid_feats[-4][0]:
        #         #featuremap = featuremap_2_heatmap(acid_feats[-3])  #for mean visua
        #         print(featuremap.shape)
        #         img = cv2.imread('data/cervix/img/02895805_2019-07-12_2.jpg')
        #         featuremap = featuremap.data.cpu().numpy()  #for per feature visua
        #         featuremap = np.maximum(featuremap, 0)      #for per feature visua
        #         featuremap /= np.max(featuremap)            #for per feature visua
        #
        #         featuremap = cv2.resize(featuremap, (img.shape[1], img.shape[0]),interpolation=cv2.INTER_NEAREST)  # 将热力图的大小调整为与原始图像相同
        #         heatmap = np.uint8(featuremap * 255) # 将热力图转换为RGB格式
        #         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        #
        #         superimposed_img = heatmap  * 0.4   + img*0.3  # 这里的0.4是热力图强度因子
        #         cv2.imwrite(os.path.join('./testoutanaly/featuremaps/2xra_dual_wh_weigg_sharebrpn_layer-4_0.4_0.3',
        #                               'featuremap_2_' + str(i) + '.png'), np.uint8(superimposed_img))  #for per feature visua
        #         #cv2.imwrite(os.path.join('./testoutanaly/featuremaps/2xra_dual_wh_weigg_sharebrpn_layer-3_featuremap'+ '.png'), np.uint8(superimposed_img)) #for mean visua
        #         i = i + 1
        #         #break #for mean visua
        #
        #     # for iodine
        #     i=0
        #     for featuremap in iodine_feats[-4][0]:
        #         #featuremap = featuremap_2_heatmap(acid_feats[-3])  #for mean visua
        #         print(featuremap.shape)
        #         img = cv2.imread('data/cervix/img/02895805_2019-07-12_3.jpg')
        #         featuremap = featuremap.data.cpu().numpy()  #for per feature visua
        #         featuremap = np.maximum(featuremap, 0)      #for per feature visua
        #         featuremap /= np.max(featuremap)            #for per feature visua
        #
        #         featuremap = cv2.resize(featuremap, (img.shape[1], img.shape[0]),interpolation=cv2.INTER_NEAREST)  # 将热力图的大小调整为与原始图像相同
        #         heatmap = np.uint8(featuremap * 255) # 将热力图转换为RGB格式
        #         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        #
        #         superimposed_img = heatmap  * 0.4   + img*0.3  # 这里的0.4是热力图强度因子
        #         cv2.imwrite(os.path.join('./testoutanaly/featuremaps/2xra_dual_wh_weigg_sharebrpn_layer-4_0.4_0.3',
        #                               'featuremap_3_' + str(i) + '.png'), np.uint8(superimposed_img))  #for per feature visua
        #         #cv2.imwrite(os.path.join('./testoutanaly/featuremaps/2xra_dual_wh_weigg_sharebrpn_layer-3_featuremap'+ '.png'), np.uint8(superimposed_img)) #for mean visua
        #         i = i + 1
        #         #break #for mean visua






        if acid_proposals is None:
            acid_proposal_list = self.rpn_head.simple_test_rpn(acid_feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is None:
            iodine_proposal_list = self.rpn_head.simple_test_rpn(iodine_feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals

        roi_result_acid = self.roi_head_acid.simple_test(acid_feats = acid_feats, iodine_feats = iodine_feats, acid_proposal_list = acid_proposal_list,
                                                         img_metas=img_metas, rescale = rescale)
        roi_result_iodine = self.roi_head_iodine.simple_test(acid_feats = iodine_feats, iodine_feats = acid_feats, acid_proposal_list = iodine_proposal_list,
                                                             img_metas=img_metas, rescale=rescale)

        return (roi_result_acid,roi_result_iodine)


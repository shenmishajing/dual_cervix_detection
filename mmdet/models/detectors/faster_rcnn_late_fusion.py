import numpy as np

from mmcv.runner import auto_fp16
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNLateFusion(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, prim = None, iou_threshold = 0.2, score_threshold = 0.8, *args, **kwargs):
        super(FasterRCNNLateFusion, self).__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        if prim is None:
            self.prim = ['acid', 'iodine']
        elif not isinstance(prim, list):
            self.prim = [prim]
        else:
            self.prim = prim

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        return super(FasterRCNNLateFusion, self).extract_feat(acid_img), super(FasterRCNNLateFusion, self).extract_feat(iodine_img)

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
        if self.with_rpn:
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

        acid_roi_losses = self.roi_head.forward_train(acid_feats, img_metas, acid_proposal_list, acid_gt_bboxes, acid_gt_labels,
                                                      acid_gt_bboxes_ignore, **kwargs)
        if 'acid' in self.prim:
            for k, v in acid_roi_losses.items():
                losses['acid_' + k] = v

        iodine_roi_losses = self.roi_head.forward_train(iodine_feats, img_metas, iodine_proposal_list, iodine_gt_bboxes, iodine_gt_labels,
                                                        iodine_gt_bboxes_ignore, **kwargs)
        if 'iodine' in self.prim:
            for k, v in iodine_roi_losses.items():
                losses['iodine_' + k] = v

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
        assert self.with_bbox, 'Bbox head must be implemented.'
        acid_feats, iodine_feats = self.extract_feat(acid_img, iodine_img)
        if acid_proposals is None:
            acid_proposal_list = self.rpn_head.simple_test_rpn(acid_feats, img_metas)
        else:
            acid_proposal_list = acid_proposals
        if iodine_proposals is None:
            iodine_proposal_list = self.rpn_head.simple_test_rpn(iodine_feats, img_metas)
        else:
            iodine_proposal_list = iodine_proposals

        acid_results = self.roi_head.simple_test(acid_feats, acid_proposal_list, img_metas, rescale = rescale)
        iodine_results = self.roi_head.simple_test(iodine_feats, iodine_proposal_list, img_metas, rescale = rescale)

        return self.fusion_results(acid_results, iodine_results)

    def fusion_results(self, acid_results, iodine_results):
        acid_res, iodine_res = [], []
        for i in range(len(acid_results)):
            cur_acid_res, cur_iodine_res = [], []
            for c in range(len(acid_results[i])):
                cur_acid_bboxes = acid_results[i][c]
                cur_acid_bboxes = cur_acid_bboxes[cur_acid_bboxes[..., -1] > self.score_threshold]
                cur_iodine_bboxes = iodine_results[i][c]
                cur_iodine_bboxes = cur_iodine_bboxes[cur_iodine_bboxes[..., -1] > self.score_threshold]
                if cur_acid_bboxes.shape[0] > 0 and cur_iodine_bboxes.shape[0] > 0:
                    iou = self.bbox_overlaps(cur_acid_bboxes, cur_iodine_bboxes)
                    cur_acid_inds = np.max(iou, axis = 1) > self.iou_threshold
                    cur_acid_res.append(cur_acid_bboxes[cur_acid_inds])
                    cur_iodine_inds = np.max(iou, axis = 0) > self.iou_threshold
                    cur_iodine_res.append(cur_iodine_bboxes[cur_iodine_inds])
                else:
                    cur_acid_res.append(cur_acid_bboxes[0:0])
                    cur_iodine_res.append(cur_iodine_bboxes[0:0])
            acid_res.append(cur_acid_res)
            iodine_res.append(cur_iodine_res)
        return acid_res, iodine_res

    @staticmethod
    def bbox_overlaps(bboxes1, bboxes2, eps = 1e-6):
        if bboxes1.shape[-1] != 4:
            bboxes1 = bboxes1[..., :4]
        if bboxes2.shape[-1] != 4:
            bboxes2 = bboxes2[..., :4]
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        lt = np.maximum(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = np.minimum(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = np.clip(rb - lt, 0, None)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap

        union = np.maximum(union, eps)
        ious = overlap / union
        return ious

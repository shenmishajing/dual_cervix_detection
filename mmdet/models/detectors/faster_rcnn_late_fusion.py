import numpy as np

from ..builder import DETECTORS
from .two_stage_cervix import TwoStageCervixDetector


@DETECTORS.register_module()
class FasterRCNNLateFusion(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, top_k = 10, iou_threshold = 0.05, score_threshold = 0.5, *args, **kwargs):
        super(FasterRCNNLateFusion, self).__init__(*args, **kwargs)
        self.top_k = top_k
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def simple_test(self, *args, **kwargs):
        return self.fusion_results(*super(FasterRCNNLateFusion, self).simple_test(*args, **kwargs))

    def fusion_results(self, acid_results, iodine_results):
        acid_res, iodine_res = [], []
        for i in range(len(acid_results)):
            cur_acid_res, cur_iodine_res = [], []
            for c in range(len(acid_results[i])):
                cur_acid_bboxes = acid_results[i][c]
                cur_iodine_bboxes = iodine_results[i][c]
                cur_acid_bboxes = cur_acid_bboxes[cur_acid_bboxes[..., -1] > self.score_threshold]
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

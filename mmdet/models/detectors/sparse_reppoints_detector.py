from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch
import torch.nn as nn 
from mmdet.core import bbox2result

@DETECTORS.register_module()
class SparseRepPointsDetector(SingleStageDetector):


    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None):

        super(SparseRepPointsDetector,
              self).__init__(backbone, 
                             neck,
                             bbox_head, 
                             train_cfg, 
                             test_cfg, 
                             pretrained)
    
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat(img)
        objectness, topn_box, topn_cls, topn_idx = self.bbox_head(x)
        # bbox_pred, cls_pred, img_metas, cfg=None, rescale=False, with_nms=False
        bbox_list = self.bbox_head.get_bboxes(
            topn_box, topn_cls, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

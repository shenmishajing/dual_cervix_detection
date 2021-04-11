import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class Faster(DualCervixTwoStageDetecor):

    def __init__(self, 
                 acid_backbone, iodine_backbone,
                 acid_neck, iodine_neck,
                 acid_rpn_head,
                 acid_tsd_head, 
                 acid_roi_head, iodine_roi_head):

        
        pass


    def init_weights(self,):


        pass


    def extract_feat(self,):

        pass


    def forward_train(self,):

        pass


    @auto_fp16(apply_to=('acid_img', 'iodine_img'))
    def forward(self, acid_img, iodine_img, img_metas, return_loss=True, **kwargs):

        if return_loss:
            return self.forward_train(acid_img, iodine_img, img_metas, **kwargs)
        else:
            return self.forward_test(acid_img, iodine_img, img_metas, **kwargs)



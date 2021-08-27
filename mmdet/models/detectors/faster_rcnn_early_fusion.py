import torch

from ..builder import DETECTORS
from .two_stage_cervix import TwoStageCervixDetector


@DETECTORS.register_module()
class FasterRCNNEarlyFusion(TwoStageCervixDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def extract_feat(self, acid_img, iodine_img):
        """Directly extract features from the backbone+neck."""
        feats = torch.cat([acid_img, iodine_img], dim = 1)
        feats = self._extract_feat(feats)
        return feats, feats

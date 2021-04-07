import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')
import cv2
import matplotlib.pyplot as plt 
""" 
{
    'filename': ['xxx_2.jpg', 'xxx_3,jpg'],
    'width': 1280,
    'height': 720,
    'ann': 
        { # [acid, iodine]
            'bboxes': [<np.ndarray, float32> (n, 4), <np.ndarray, float32> (n, 4)],
            'labels': [<np.ndarray, int64> (n, ), <np.ndarray, int64> (n, ),]
            'bboxes_ignore': [<np.ndarray, float32> (k, 4), <np.ndarray, float32> (k, 4),]
            'labels_ignore': [<np.ndarray, int64> (k, ) (optional field), <np.ndarray, int64> (k, ) (optional field)]
        },
},
"""

@DATASETS.register_module()
class DualCervixDataset(CustomDataset):

    CLASSES = ('lsil', 'hsil')

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[list[int]]: All categories in the acid cervix image and 
            iodine cervix image of specified index.
        """

        return [x.astype(np.int).tolist() for x in self.data_infos[idx]['ann']['labels']]


    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['bbox_fields'] = []
        results['label_fields'] = []


    def vis_for_check(self, data):
        #! 可视化之前时，将normlization屏蔽掉
        img_prefix = "/data/luochunhua/od/mmdetection/data/cervix/img"
        ori_acid_fname, ori_iodine_fname = data["img_metas"].data["filename"]
        ori_acid_path = osp.join(img_prefix, ori_acid_fname)
        ori_iodine_path = osp.join(img_prefix, ori_iodine_fname)

        ori_aicd_img = cv2.imread(ori_acid_path)
        # ori_aicd_img = cv2.cvtColor(ori_aicd_img, cv2.COLOR_BGR2RGB)
        ori_iodine_img = cv2.imread(ori_iodine_path)
        # ori_iodine_img = cv2.cvtColor(ori_iodine_img, cv2.COLOR_BGR2RGB)
        
        output_dir = "/data/luochunhua/od/mmdetection/test_output"
        ori_dst_acid_path = osp.join(output_dir, ori_acid_fname)
        ori_dst_iodine_path = osp.join(output_dir, ori_iodine_fname)
        cv2.imwrite(ori_dst_acid_path, ori_aicd_img)
        cv2.imwrite(ori_dst_iodine_path, ori_iodine_img)

        acid_img = data["acid_img"].data.numpy().transpose((1,2,0)).astype(np.uint8)
        iodine_img = data["iodine_img"].data.numpy().transpose((1,2,0)).astype(np.uint8)
        acid_gt_bboxes = data["acid_gt_bboxes"].data.numpy().astype(np.int32)
        iodine_gt_bboxes = data['iodine_gt_bboxes'].data.numpy().astype(np.int32)
        acid_gt_labels = data["acid_gt_labels"].data.numpy()
        iodine_gt_labels = data['iodine_gt_labels'].data.numpy()

        tmp_acid_img = cv2.UMat(acid_img).get()
        for box, label in zip(acid_gt_bboxes, acid_gt_labels):
            tmp_acid_img = cv2.rectangle(tmp_acid_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

        tmp_iodine_img = cv2.UMat(iodine_img).get()
        for box, label in zip(iodine_gt_bboxes, iodine_gt_labels):
            tmp_iodine_img = cv2.rectangle(tmp_iodine_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

        dst_acid_path = osp.join(output_dir, "t_" + ori_acid_fname)
        dst_iodine_path = osp.join(output_dir, "t_" + ori_iodine_fname)

        cv2.imwrite(dst_acid_path, tmp_acid_img)
        cv2.imwrite(dst_iodine_path, tmp_iodine_img)


    def evaluate(self, 
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):

        pass
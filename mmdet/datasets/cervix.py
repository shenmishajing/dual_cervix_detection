import copy 
import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
import pickle 

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from .pipelines import Compose
from .coco import CocoDataset

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

class CervixDataset(CocoDataset):
    #! 单、双模态dataset都要用到相同的评价，所以写了这个父类来完成指标评价部分

    def convert_dets_format(self, dets):
        """ 
            检测结果的格式：
                img1_list = [arr_cls1, arr_cls2, ...], 
                    某个类别为空的arr = np.zeros((0, 5), float32)
                    不空的时候为 arr = np.([
                                            [xmin, ymin, xmax, ymax, score],
                                            [], ...])
                dets = [img1_list, img2_list, ...]
            
            目标格式：
                {
                    cls1_ind: list[dict],
                    cls2_ind: list[dict],
                    ...
                }

                dict: { 每个检测框一个字典
                    "image_id": image_id,
                    "box": , np.array([xmin, ymin, xmax, ymax], np.float32)
                    "score": 0.65
                }
        """
        tf_dets = defaultdict(list)
        for image_id, det in enumerate(dets):
            for cls_ind, arr in enumerate(det):
                for j in range(arr.shape[0]):
                    tf_dets[cls_ind].append({
                        "image_id": image_id,
                        "box": arr[j, :4],
                        "score": float(arr[j, 4])
                    })

        return tf_dets


    def get_format_annos(self, data_infos):
        """ 
            gts原本的格式：
                COCO
                self.data_infos[idx] = {
                    'file_name': '08274633_2016-05-11_2.jpg',
                    'height': 600, 
                    'width': 733, 
                    'id': 6256, 
                    'filename': '08274633_2016-05-11_2.jpg'
                }
                anno = self.get_ann_info(idx)
                anno = dict(
                    bboxes=gt_bboxes, np.array float32
                    labels=gt_labels, np.array int64
                    bboxes_ignore=gt_bboxes_ignore,
                    masks=gt_masks_ann,
                    seg_map=seg_map)

            目标格式：
                {
                    image_id: [{
                        "class": cls_id,
                        "box": [xmin, ymin, xmax, ymax]
                    }, {
                        "class": cls_id,
                        "box": [xmin, ymin, xmax, ymax]
                    }, ...]
                    ...
                }
        """
        tf_annos = defaultdict(list)
        for idx in range(len(data_infos)):
            image_id = idx
            anno = self.get_ann_info(idx)  
            for box, label in zip(anno["bboxes"].tolist(), anno["labels"].tolist()):
                tf_annos[image_id].append({
                    "class": label,
                    "box": [int(round(x)) for x in box]
                })

        return tf_annos


    def evaluate_single(self, predictions, annos, suffix=''):
        predictions = self.sort_predictions(predictions)
        K = len(self._class_names)  # class
        T = len(self._iou_threshs)  # iou thresh
        M = len(self._max_dets)  # max detection per image
        aps = -np.ones((K, T, M))
        ars = -np.ones((K, T, M))
        frocs = -np.ones((K, T, M))
        rec_img_list = -np.ones((K, T, M))
        for k_i, cls_name in enumerate(self._class_names):
            if k_i not in predictions:
                continue
            dts = predictions[k_i]
            gts = self.get_cls_gts(annos, k_i)
            for t_i, thresh in enumerate(self._iou_threshs):  # iou from 0.5 to 0.95, step 0.05
                for m_i, max_det in enumerate(self._max_dets):
                    max_rec, ap, froc, rec_img = self.eval(dts, gts, ovthresh=thresh / 100, max_det=max_det)
                    aps[k_i, t_i, m_i] = ap * 100
                    ars[k_i, t_i, m_i] = max_rec * 100
                    frocs[k_i, t_i, m_i] = froc * 100
                    rec_img_list[k_i, t_i, m_i] = rec_img * 100

        self._result = {
            'aps' + suffix: aps,
            'ars' + suffix: ars,
            'frocs' + suffix: frocs,
            'rec_img_list' + suffix: rec_img_list
        }

        record = self.summarize(suffix)

        return record


    def summarize(self, suffix):
        def _summarize(type, iou_t=None, max_det=100):
            suffix_output_str = suffix if suffix == '' else f' {suffix[1:]}'  # '_acid' to ' acid'
            i_str = ' {:<25} @[ IoU={:<9} | maxDets={} ] = {:0.5f}'
            mind = [i for i, mdet in enumerate(self._max_dets) if mdet == max_det]
            if iou_t is None:
                tind = slice(len(self._iou_threshs))
            else:
                tind = [i for i, iou_thresh in enumerate(self._iou_threshs) if iou_thresh == iou_t]
            iou_str = '{:0.2f}:{:0.2f}'.format(0.5, 0.95) if iou_t is None else '{:0.2f}'.format(iou_t)
            det_str = '{:>3d}'.format(max_det) if max_det != float('inf') else 'all'

            if type == 'ap':
                title_str = 'Average Precision'
                metric_res = np.mean(self._result['aps' + suffix][:, tind, mind])
            elif type == 'ar':
                title_str = 'Average Recall'
                metric_res = np.mean(self._result['ars' + suffix][:, tind, mind])
            elif type == 'froc':
                title_str = 'FROC'
                metric_res = np.mean(self._result['frocs' + suffix][:, tind, mind])
            elif type == 'irec':
                title_str = 'Image Recall'
                metric_res = np.mean(self._result['rec_img_list' + suffix][:, tind, mind])
            else:
                raise ValueError
            title_str += suffix_output_str
            metric_res = float(metric_res)
            # self._logger.info(i_str.format(title_str, iou_str, det_str, metric_res))
            return metric_res

        ret = OrderedDict()
        # ap
        for max_det in self._max_dets:
            ret[f'AP_Top{max_det}' + suffix] = _summarize(type='ap', iou_t=None, max_det=max_det)
            ret[f'AP50_Top{max_det}' + suffix] = _summarize(type='ap', iou_t=50, max_det=max_det)
            ret[f'AP75_Top{max_det}' + suffix] = _summarize(type='ap', iou_t=75, max_det=max_det)
        # ar
        for max_det in self._max_dets:
            ret[f'AR_Top{max_det}' + suffix] = _summarize(type='ar', iou_t=None, max_det=max_det)
        # froc
        ret[f'FROC' + suffix] = _summarize(type='froc', iou_t=None, max_det=self._max_dets[-1])
        ret[f'FROC50' + suffix] = _summarize(type='froc', iou_t=50, max_det=self._max_dets[-1])
        ret[f'FROC75' + suffix] = _summarize(type='froc', iou_t=75, max_det=self._max_dets[-1])

        # image level recall
        for max_det in self._max_dets:
            ret[f'iRecall_Top{max_det}' + suffix] = _summarize(type='irec', iou_t=None, max_det=max_det)
            ret[f'iRecall50_Top{max_det}' + suffix] = _summarize(type='irec', iou_t=50, max_det=max_det)
            ret[f'iRecall75_Top{max_det}' + suffix] = _summarize(type='irec', iou_t=75, max_det=max_det)
        # print(ret)
        return ret


    @staticmethod
    def sort_predictions(predictions):
        """
        sort each class's predictions in score descending order, preserve only image_id and detection boxes
        """
        res = {}
        for cls_id, cls_predictions in predictions.items():
            image_ids = [x['image_id'] for x in cls_predictions]
            scores = np.array([x['score'] for x in cls_predictions])
            boxes = np.array([x['box'] for x in cls_predictions]).astype(float)
            # sort by score
            sorted_ind = np.argsort(-scores)
            dt_boxes = boxes[sorted_ind, :]
            dt_image_ids = [image_ids[x] for x in sorted_ind]
            res[cls_id] = {
                'dt_boxes': dt_boxes,
                'dt_image_ids': dt_image_ids
            }

        return res


    @staticmethod
    def get_cls_gts(annos, cls_id):
        # get this class's gt
        gt_recs = {}
        npos = 0
        for image_id, annos in annos.items():
            R = [x for x in annos if x['class'] == cls_id]
            boxes = np.array([x['box'] for x in R]).astype(float)
            det = [False] * len(R)
            npos += len(R)
            gt_recs[image_id] = {'boxes': boxes, 'det': det}
        return {'gt_recs': gt_recs, 'npos': npos}


    @staticmethod
    def eval(dts, gts, ovthresh=0.5, max_det=np.inf):
        """
        eval one class
        """
        gt_recs = copy.deepcopy(gts['gt_recs'])
        dt_boxes = dts['dt_boxes']
        dt_image_ids = dts['dt_image_ids']

        # go down dts and mark TPs and FPs
        # recall = tp / npos
        # preicision = tp / tp + fp
        # ap = roc of P-R
        # froc = sum(recall_i) when fp per img = 1/8, 1/4, 1/2, 1, 2, 4, 8
        # fp_per_img = fp / nimg
        max_det_count = defaultdict(int)
        nimg = len(gt_recs)
        img_m = np.zeros(nimg)  # calculate recall on image level, means patient level recall
        fps_thresh = nimg * np.array([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8])
        npos = gts['npos']
        tp = []  # true positive, for recall and precision
        fp = []  # false positive, for precision
        for i in range(len(dt_image_ids)):
            image_id = dt_image_ids[i]
            if max_det_count[image_id] >= max_det:
                continue
            max_det_count[image_id] += 1
            dt_box = dt_boxes[i]
            R = gt_recs[image_id]
            gt_boxes = R['boxes']
            ovmax = -np.inf

            if gt_boxes.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(gt_boxes[:, 0], dt_box[0])
                iymin = np.maximum(gt_boxes[:, 1], dt_box[1])
                ixmax = np.minimum(gt_boxes[:, 2], dt_box[2])
                iymax = np.minimum(gt_boxes[:, 3], dt_box[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                # union
                uni = (
                        (dt_box[2] - dt_box[0] + 1.0) * (dt_box[3] - dt_box[1] + 1.0)
                        + (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0)
                        - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:  # match
                if not R['det'][jmax]:  # gt hasn't been matched
                    img_m[image_id] = 1
                    tp.append(1.0)
                    fp.append(0.0)
                    R['det'][jmax] = 1
                else:  # gt has been matched
                    tp.append(0.0)
                    fp.append(1.0)
            else:  # not match
                tp.append(0.0)
                fp.append(1.0)

        # compute precision recall
        tp = np.array(tp)
        fp = np.array(fp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = SingleCervixDataset.voc_ap(rec, prec)

        # find first idx where fp > fp_thresh, append sentinel values at the end
        fp = np.concatenate((fp, [np.inf]))
        fp_idx = [min((fp > x).nonzero()[0][0], len(fp) - 2) for x in fps_thresh]
        froc = np.mean([rec[idx] for idx in fp_idx])
        max_rec = rec.max()
        rec_img = np.sum(img_m) / len(img_m)

        return max_rec, ap, froc, rec_img


    @staticmethod
    def voc_ap(rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap


@DATASETS.register_module()
class DualCervixDataset(CervixDataset):
    """ 
    最先好看看CocoDataset和CustomDataset，理解一下数据是怎么加载、处理的
    由acid_coco, iodine_coco构成数据集
    """
    CLASSES = ('lsil', 'hsil')

    def __init__(self,
                 prim,
                 acid_ann_file,
                 iodine_ann_file,
                 pipeline,
                 classes,
                 data_root=None,
                 img_prefix='',
                 proposal_file=None,                
                 test_mode=False,
                 filter_empty_gt=True):
        """

        Args:
            prim (str): "acid", "iodine", None. prim="acid", 表示用iodine来辅助acid.主要影响下面的evaluation。
                检测模型的返回的检测结果是把主模态放在前面，辅助模态的结果放在后面[prim_result_1, acid_result_1, prim_result_2, acid_result_2, ...]
                当前模型是醋酸为主模态碘为辅助模态时，醋酸的结果放在前面，碘的结果放在后面
            acid_ann_file ([type]): 醋酸的标注文件
            iodine_ann_file ([type]): 碘的标注文件
            pipeline ([type]): 数据增强
            classes ([type]): 类别名
            data_root ([type], optional): [description]. Defaults to None.
            img_prefix (str, optional): [description]. Defaults to ''.
            proposal_file ([type], optional): [description]. Defaults to None.
            test_mode (bool, optional): [description]. Defaults to False.
            filter_empty_gt (bool, optional): [description]. Defaults to True.
        """
        assert prim in ('acid', 'iodine', None)
        #! single_gpu_test 依赖prim
        self.prim = prim
        self.acid_ann_file = acid_ann_file
        self.iodine_ann_file = iodine_ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.acid_ann_file):
                self.acid_ann_file = osp.join(self.data_root, self.acid_ann_file)
            if not osp.isabs(self.iodine_ann_file):
                self.iodine_ann_file = osp.join(self.data_root, self.iodine_ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)

        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.acid_ann_file, self.iodine_ann_file)
        
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # TODO
        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

        self._iou_threshs = list(range(50, 100, 5))
        self._max_dets = [1, 2, 3, 5, 10, 100] 
        self._class_names = self.get_classes(classes=classes)


    def load_annotations(self, acid_ann_file, iodine_ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.acid_coco = COCO(acid_ann_file)
        self.iodine_coco = COCO(iodine_ann_file)

        self.cat_ids = self.acid_coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.acid_coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.acid_coco.load_imgs([i])[0]
            info['filename'] = [info['file_name'], info['file_name'].replace("2.jpg", "3.jpg")]
            data_infos.append(info)
        
        return data_infos


    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        acid_ann_ids = self.acid_coco.get_ann_ids(img_ids=[img_id])
        acid_ann_info = self.acid_coco.load_anns(acid_ann_ids)
        # print(acid_ann_info)
        iodine_ann_ids = self.iodine_coco.get_ann_ids(img_ids=[img_id])
        iodine_ann_info = self.iodine_coco.load_anns(iodine_ann_ids)
        # print(iodine_ann_info)
        acid_ann = self._parse_ann_info(self.data_infos[idx], acid_ann_info)
        iodine_ann = self._parse_ann_info(self.data_infos[idx], iodine_ann_info)

        ann = dict(
            bboxes=[acid_ann["bboxes"], iodine_ann["bboxes"]],
            labels=[acid_ann["labels"], iodine_ann["labels"]])

        return ann 


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                
        box_idx = None
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            
            #! 确保醋酸和碘的框的排列顺序是一致的（标注时，已经确定了对应关系，只要按左上角顶点的x坐标值进行排序，即可对应）
            box_idx = np.argsort(gt_bboxes[:, 0])
            gt_bboxes = gt_bboxes[box_idx]
            gt_labels = gt_labels[box_idx]
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)


        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels)

        return ann


    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        acid_ann_ids = self.acid_coco.get_ann_ids(img_ids=[img_id])
        acid_ann_info = self.acid_coco.load_anns(acid_ann_ids)

        iodine_ann_ids = self.iodine_coco.get_ann_ids(img_ids=[img_id])
        iodine_ann_info = self.iodine_coco.load_anns(iodine_ann_ids)

        return [
            [ann['category_id'] for ann in acid_ann_info],
            [ann['category_id'] for ann in iodine_ann_info]
        ]


    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        acid_ids_with_ann = set(_['image_id'] for _ in self.acid_coco.anns.values())
        iodine_ids_with_ann = set(_['image_id'] for _ in self.iodine_coco.anns.values())
        ids_with_ann = acid_ids_with_ann & iodine_ids_with_ann

        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.acid_coco.cat_img_map[class_id])
            ids_in_cat |= set(self.iodine_coco.cat_img_map[class_id])

        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds


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


    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        else:
            raise TypeError('invalid type of results')
        return result_files


    def format_results(self, results, jsonfile_prefix=None, acid=True, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        suffix = "acid" if acid else "iodine"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results_{}'.format(suffix))
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir


    def evaluate_coco(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """
        #! 双检测的结果[prim_result, aux_result, prim_result, aux_result, ....]
        #! coco 中的指标计算方法，搬过来，修改适应双模态的评估(COCO的评价指标)。这是最一开始的评价指标，已经不用了。
        #! 现在用的是CervixDataset中指标计算方法，就是那一大串指标。
        Args:
            results ([type]): [description]
            metric (str, optional): [description]. Defaults to 'bbox'.
            logger ([type], optional): [description]. Defaults to None.
            jsonfile_prefix ([type], optional): [description]. Defaults to None.
            classwise (bool, optional): [description]. Defaults to False.
            proposal_nums (tuple, optional): [description]. Defaults to (100, 300, 1000).
            iou_thrs ([type], optional): [description]. Defaults to None.
            metric_items ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        prim_results = []
        aux_results = []
        for i in range(len(results) // 2):
            prim_results.extend(results[2 * i])
            aux_results.extend(results[2 * i + 1])    

        if self.prim == "acid":
            acid_results = prim_results
            iodine_results = aux_results 
        else:
            acid_results = aux_results
            iodine_results = prim_results

        acid_metric = self.evaluate_single_coco(acid_results, 
                            acid= self.prim == "acid",
                            metric=metric, 
                            logger=logger,
                            jsonfile_prefix=jsonfile_prefix,
                            classwise=classwise,
                            proposal_nums=proposal_nums,
                            iou_thrs=iou_thrs,
                            metric_items=metric_items)

        iodine_metric = self.evaluate_single_coco(iodine_results, 
                            acid= self.prim != "acid",
                            metric=metric, 
                            logger=logger,
                            jsonfile_prefix=jsonfile_prefix,
                            classwise=classwise,
                            proposal_nums=proposal_nums,
                            iou_thrs=iou_thrs,
                            metric_items=metric_items)

        ret = dict()
        ret.update({k + "_acid": v for k,v in acid_metric.items()})
        ret.update({k + "_iodine": v for k,v in iodine_metric.items()})

        return ret


    def evaluate_single_coco(self,
                 results,
                 acid=True,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix, acid)

        eval_results = OrderedDict()
        cocoGt = eval("self.{}_coco".format("acid" if acid else "iodine"))
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results   


    def get_format_annos(self, data_infos):
        acid_tf_annos = defaultdict(list)
        iodine_tf_annos = defaultdict(list)

        for idx in range(len(data_infos)):
            image_id = idx
            anno = self.get_ann_info(idx) 

            for box, label in zip(anno["bboxes"][0].tolist(), anno["labels"][0].tolist()):
                acid_tf_annos[image_id].append({
                    "class": label,
                    "box": [int(round(x)) for x in box]
                })

            for box, label in zip(anno["bboxes"][1].tolist(), anno["labels"][1].tolist()):
                iodine_tf_annos[image_id].append({
                    "class": label,
                    "box": [int(round(x)) for x in box]
                })

        return acid_tf_annos, iodine_tf_annos


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """将醋酸和碘的检测结果分离出来，分别计算评价指标

        Args:
            results (list): 检测模型返回出来的结果, [prim_result_1, acid_result_1, prim_result_2, acid_result_2, ...]
            下面的参数没用，不用管。本来是cocodataset接口中参数
            metric (str, optional): [description]. Defaults to 'bbox'.
            logger ([type], optional): [description]. Defaults to None.
            jsonfile_prefix ([type], optional): [description]. Defaults to None.
            classwise (bool, optional): [description]. Defaults to False.
            proposal_nums (tuple, optional): [description]. Defaults to (100, 300, 1000).
            iou_thrs ([type], optional): [description]. Defaults to None.
            metric_items ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        prim_results = []
        aux_results = []
        for i in range(len(results) // 2):
            prim_results.extend(results[2 * i])
            aux_results.extend(results[2 * i + 1])

        acid_results = prim_results if self.prim == "acid" else aux_results
        iodine_results = aux_results if self.prim == "acid" else prim_results

        acid_tf_dets = self.convert_dets_format(acid_results)
        iodine_tf_dets = self.convert_dets_format(iodine_results)

        acid_tf_annos, iodine_tf_annos = self.get_format_annos(self.data_infos)

        acid_ret = super(DualCervixDataset, self).evaluate_single(acid_tf_dets, acid_tf_annos, "_acid")
        iodine_ret = super(DualCervixDataset, self).evaluate_single(iodine_tf_dets, iodine_tf_annos, "_iodine")

        ret = dict()
        ret.update(acid_ret)
        ret.update(iodine_ret)
        return ret
  

@DATASETS.register_module()
class SingleCervixDataset(CervixDataset):
    #! 单模态模型本来可以直接采用默认的cocodataset，但是由于必须用指定的评价指标就重新写了这个类，与cocodataset只是指标计算部分不同
    CLASSES = ('lsil', 'hsil')
    
    def __init__(self,
                 img_type,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        super(SingleCervixDataset, self).__init__(ann_file, 
                                                  pipeline, 
                                                  classes, 
                                                  data_root, 
                                                  img_prefix, 
                                                  seg_prefix, 
                                                  proposal_file, 
                                                  test_mode, 
                                                  filter_empty_gt)
        #! img_type 取值为acid, iodine，用来加到评估指标的命名中
        self.img_type = img_type
        self._iou_threshs = list(range(50, 100, 5))
        self._max_dets = [1, 2, 3, 5, 10, 100] 
        self._class_names = self.get_classes(classes=classes)


    def evaluate(self, 
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=None,
                 metric_items=None):
        # tt = super(SingleCervixDataset, self).evaluate(results,metric,logger,jsonfile_prefix,classwise,proposal_nums,iou_thr,metric_items)
        tf_dets = self.convert_dets_format(results)      
        tf_annos = self.get_format_annos(self.data_infos)
        ret = self.evaluate_single(tf_dets, tf_annos, suffix="_" + self.img_type)        

        return ret
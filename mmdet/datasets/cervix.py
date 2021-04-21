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
from .pipelines import Compose

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
    """ 
    由acid_coco, iodine_coco构成数据集
    """
    CLASSES = ('lsil', 'hsil')

    def __init__(self,
                prim,
                acid_ann_file,
                iodine_ann_file,
                pipeline,
                classes,
                dual_det=False,
                data_root=None,
                img_prefix='',
                proposal_file=None,                
                test_mode=False,
                filter_empty_gt=True):
        
        assert prim in ('acid', 'iodine', None)

        self.prim = prim
        self.dual_det = dual_det
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

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
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


    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
    

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results
    
    
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


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        if self.dual_det:
            #! 双检测的结果[prim_result, aux_result, prim_result, aux_result, ....]
            prim_results = []
            aux_results = []
            for i in range(len(results) // 2):
                prim_results.append(results[2 * i])
                aux_results.append(results[2 * i + 1])

            if self.prim == "acid":
                prim = "acid"
                aux = "iodine"
            else:
                prim = "iodine"
                aux = "acid"               

            prim_metric = self.evaluate_single(prim_results, 
                                acid= self.prim == "acid",
                                metric=metric, 
                                logger=logger,
                                jsonfile_prefix=jsonfile_prefix,
                                classwise=classwise,
                                proposal_nums=proposal_nums,
                                iou_thrs=iou_thrs,
                                metric_items=metric_items)

            aux_metric = self.evaluate_single(aux_results, 
                                acid= self.prim != "acid",
                                metric=metric, 
                                logger=logger,
                                jsonfile_prefix=jsonfile_prefix,
                                classwise=classwise,
                                proposal_nums=proposal_nums,
                                iou_thrs=iou_thrs,
                                metric_items=metric_items)
            ret = {
                "prim({})".format(prim): prim_metric,
                "aux({})".format(aux): aux_metric
                }
            
        else:
            if self.prim == "acid":
                prim_metric = self.evaluate_single(results, 
                                    acid=True,
                                    metric=metric, 
                                    logger=logger,
                                    jsonfile_prefix=jsonfile_prefix,
                                    classwise=classwise,
                                    proposal_nums=proposal_nums,
                                    iou_thrs=iou_thrs,
                                    metric_items=metric_items)
                ret = {
                    "prim(acid)": prim_metric}
            else:
                iodine_metric = self.evaluate_single(results, 
                                    acid=False,
                                    metric=metric, 
                                    logger=logger,
                                    jsonfile_prefix=jsonfile_prefix,
                                    classwise=classwise,
                                    proposal_nums=proposal_nums,
                                    iou_thrs=iou_thrs,
                                    metric_items=metric_items)
                ret = {
                    "prim(iodine)": iodine_metric}
            
        
        return ret


    def evaluate_single(self,
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
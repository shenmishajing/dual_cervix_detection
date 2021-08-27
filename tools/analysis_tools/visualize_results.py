import os.path as osp
import random
import argparse

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset, get_loading_pipeline
from mmdet.core.visualization import imshow_gt_det_bboxes, imshow_det_bboxes

show_gt_det_bbox_kwargs = dict(gt_bbox_color = 'green', gt_text_color = 'white', det_bbox_color = 'red', det_text_color = 'white',
                               thickness = 5, font_size = 60, show = False)
show_gt_bbox_kwargs = dict(bbox_color = show_gt_det_bbox_kwargs['gt_bbox_color'],
                           text_color = show_gt_det_bbox_kwargs['gt_text_color'],
                           **{k: v for k, v in show_gt_det_bbox_kwargs.items() if 'color' not in k})


def get_topk_results(results, topk):
    for i in range(len(results)):
        results[i] = results[i][results[i][:, -1].argsort()[::-1]]
    inds = [0 for _ in range(len(results))]
    for k in range(topk):
        for m in range(len(inds)):
            if len(results[m]) > inds[m]:
                break
        c = m + 1
        while c < len(inds):
            if results[c][inds[c]:-1] > results[m][inds[m]:-1]:
                m = c
            c += 1
        if m < len(inds):
            inds[m] += 1
        else:
            break
    for i in range(len(results)):
        results[i] = results[i][:inds[i], ]
    return results


def visualize_bboxes(filename, img, gt_bboxes, gt_labels, class_names, show_dir, result = None, score_thr = None, suffix = None,
                     visualize_num_match_gt = None):
    # calc save file path
    fname, name = osp.splitext(osp.basename(filename))
    save_filename = fname + (('_' + suffix) if suffix else '') + name
    out_file = osp.join(show_dir, save_filename)
    if result is None:
        imshow_det_bboxes(
            img,
            gt_bboxes,
            gt_labels,
            None,
            class_names,
            out_file = out_file,
            **show_gt_bbox_kwargs)
    else:
        visualize_num = len(gt_bboxes)
        if not visualize_num_match_gt:
            visualize_num += random.randint(-3, 3)
        result = get_topk_results(result, visualize_num)
        imshow_gt_det_bboxes(
            img,
            dict(gt_bboxes = gt_bboxes, gt_labels = gt_labels),
            result,
            class_names,
            score_thr = score_thr,
            out_file = out_file,
            **show_gt_det_bbox_kwargs)


# mmdetection
def visualize_results(dataset,
                      results = None,
                      score_thr = 0,
                      visualize_num_match_gt = False,
                      show_dir = 'work_dir',
                      suffix = ''):
    """Evaluate and show results.

    Args:
        dataset (Dataset): A PyTorch dataset.
        results (list): Det results from test results pkl file
        show_dir (str, optional): The filename to write the image.
            Default: 'work_dir'
    """
    mmcv.mkdir_or_exist(show_dir)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i in range(len(dataset)):
        # self.dataset[i] should not call directly
        # because there is a risk of mismatch
        data_info = dataset.prepare_train_img(i)
        if hasattr(dataset, 'prim'):
            prim = dataset.prim
            aux = ''
            for img_name in data_info['img_fields']:
                if prim not in img_name:
                    aux = img_name[:-4]
            if not aux:
                raise RuntimeError('can not find aux name')

            for stage_num, stage in enumerate([prim, aux]):
                visualize_bboxes(data_info['filename'][stage_num], data_info[stage + '_img'], data_info[stage + '_gt_bboxes'],
                                 data_info[stage + '_gt_labels'], dataset.CLASSES, show_dir,
                                 None if results is None else results[i * 2 + stage_num][0], score_thr, suffix, visualize_num_match_gt)
        else:
            visualize_bboxes(data_info['filename'], data_info['img'], data_info['gt_bboxes'], data_info['gt_labels'], dataset.CLASSES,
                             show_dir, None if results is None else results[i], score_thr, suffix, visualize_num_match_gt)
        prog_bar.update()


# detectron2
def visualize_detectron2_results(dataset,
                                 results = None,
                                 score_thr = 0,
                                 visualize_num_match_gt = False,
                                 show_dir = 'work_dir',
                                 suffix = ''):
    """Evaluate and show results.

    Args:
        dataset (Dataset): A PyTorch dataset.
        results (list): Det results from test results pkl file
        show_dir (str, optional): The filename to write the image.
            Default: 'work_dir'
    """
    mmcv.mkdir_or_exist(show_dir)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i in range(len(dataset)):
        # self.dataset[i] should not call directly
        # because there is a risk of mismatch
        data_info = dataset.prepare_train_img(i)
        filename = data_info['filename'].split('/')[-1].split('.')[0]
        visualize_bboxes(data_info['filename'], data_info['img'], data_info['gt_bboxes'], data_info['gt_labels'], dataset.CLASSES, show_dir,
                         None if results is None else results[filename], score_thr, suffix, visualize_num_match_gt)
        prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(
        description = 'MMDet visualize an output')
    parser.add_argument('config', help = 'test config file path')
    parser.add_argument(
        '--prediction-path', help = 'prediction path where test pkl result')
    parser.add_argument(
        '--visualize-dir', help = 'directory where visualized images will be saved')
    parser.add_argument(
        '--visualize-score-thr',
        type = float,
        default = 0.3,
        help = 'score threshold (default: 0.3)')
    parser.add_argument(
        '--visualize-gt-only',
        action = 'store_true',
        help = 'only visualize ground truth')
    parser.add_argument(
        '--visualize-num-match-gt',
        action = 'store_true',
        help = 'only visualize num of det bboxes as ground truth')
    parser.add_argument(
        '--detectron2',
        action = 'store_true',
        help = 'visualize detectron2 results')
    parser.add_argument(
        '--cfg-options',
        nargs = '+',
        action = DictAction,
        help = 'override some settings in the used config, the key-value pair '
               'in xxx=yyy format will be merged into config file. If the value to '
               'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
               'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
               'Note that the quotation marks are necessary and that no white space '
               'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    if args.detectron2:
        if args.prediction_path:
            outputs = mmcv.load(args.prediction_path)
            suffix = '_'.join(args.prediction_path.split('/')[-3:-1])
        else:
            outputs = None
            suffix = 'groud_truth'
        visualize_detectron2_results(dataset, outputs, show_dir = args.visualize_dir, score_thr = args.visualize_score_thr,
                                     visualize_num_match_gt = args.visualize_num_match_gt, suffix = suffix)
    else:
        if args.prediction_path:
            outputs = mmcv.load(args.prediction_path)
            suffix = '_'.join(args.prediction_path.split('/')[-2:]).split('.')[0]
        else:
            outputs = None
            suffix = 'groud_truth'
        visualize_results(dataset, outputs, show_dir = args.visualize_dir, score_thr = args.visualize_score_thr,
                          visualize_num_match_gt = args.visualize_num_match_gt, suffix = suffix)


if __name__ == '__main__':
    main()

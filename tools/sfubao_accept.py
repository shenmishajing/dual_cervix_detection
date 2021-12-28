import argparse
import torch
import numpy as np
import os
from PIL import Image
from flask import Flask, render_template, request
import requests
from io import BytesIO
import json
from mmcv import Config, DictAction
import warnings
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor, get_loading_pipeline)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.core.visualization import imshow_det_bboxes
import mmcv
from mmcv.image import tensor2imgs
import cv2
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default='/data2/hhp/model/cervix_project/atss_r101_fpn_1x_iodine_hsil_totalanno/atss_r101_fpn_1x_iodine_hsil_totalanno.py',
                        metavar="FILE",
                        help='test config file path')
    parser.add_argument('--checkpoint', default='/data2/hhp/model/cervix_project/atss_r101_fpn_1x_iodine_hsil_totalanno/epoch_9.pth', help='checkpoint file')

    # parser.add_argument(
    #     '--visualize-dir', help='directory where gt and pred boxes visualized images will be saved')  # add for visual pred and gt by hhp
    # parser.add_argument(
    #     '--visualize-num-match-gt',
    #     action='store_true',
    #     help='only visualize num of det bboxes as ground truth') ## add for visual pred and gt by hhp
    parser.add_argument(
        '--visualize-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)') ## add for visual pred and gt by hhp

    # parser.add_argument(
    #     '--work-dir',
    #     help='the directory to save the file containing evaluation metrics')
    #parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    # parser.add_argument(
    #     '--format-only',
    #     action='store_true',
    #     help='Format the output results without perform evaluation. It is'
    #     'useful when you want to format the result to a specific format and '
    #     'submit it to the test server')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
    #     ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    #parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument(
    #     '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    # parser.add_argument(
    #     '--gpu-collect',
    #     action='store_true',
    #     help='whether to use gpu to collect results.')
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #     'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def url_loader(url):
    #r = requests.get(url)
    #use cv2
    # image = cv2.imread(BytesIO(r.content), cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    #use Image
    #image = Image.open(BytesIO(r.content)).convert('RGB')
    image = Image.open(url).convert('RGB')

    #image.save("./iodine_ori_img.jpg")
    return image

def pre_pipeline(results):
    """Prepare results dict for pipeline."""
    results['img_prefix'] = '/data2/hhp/online/cervix_detect/savemid/'
    results['seg_prefix'] = None
    results['proposal_file'] = None
    results['bbox_fields'] = []
    results['mask_fields'] = []
    results['seg_fields'] = []
    return results

def iou_(all_boxes,dt_box,iou_th=0.5):
    ixmin = np.maximum(all_boxes[:, 0], dt_box[0])
    iymin = np.maximum(all_boxes[:, 1], dt_box[1])
    ixmax = np.minimum(all_boxes[:, 2], dt_box[2])
    iymax = np.minimum(all_boxes[:, 3], dt_box[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
            (dt_box[2] - dt_box[0] + 1.0) * (dt_box[3] - dt_box[1] + 1.0)
            + (all_boxes[:, 2] - all_boxes[:, 0] + 1.0) * (all_boxes[:, 3] - all_boxes[:, 1] + 1.0)
            - inters
    )
    overlaps = inters / uni
    return all_boxes[overlaps<iou_th]


def iou_delete(order_result,iou_th=0.5):
    #将order_result迭代计算，每次计算返回与首位box的iou低于0.5的框和首位框。且由于取top3个置信度的框，因此迭代3次即可
    last_result = order_result[0].reshape(1, -1)
    new_result = iou_(order_result[1:], order_result[0], iou_th=iou_th)
    last_result = np.vstack((last_result, new_result[0].reshape(1, -1)))
    new_result = iou_(new_result[1:], new_result[0], iou_th=iou_th)
    last_result = np.vstack((last_result, new_result[0].reshape(1, -1)))
    new_result = iou_(new_result[1:], new_result[0], iou_th=iou_th)
    last_result = np.vstack((last_result, new_result))

    return last_result



def comput(img,pic_name,min_size=32):
    height,width=img.shape[0],img.shape[1]
    assert min(height,width) >= min_size
    img_info={'file_name':pic_name,'height':height,'width':width,'id':0,'filename':pic_name}
    results = dict(img_info=img_info)
    results = pre_pipeline(results)
    data = pipeline(results)
    #img = data['img'][0]
    #print('image array sum 2:', np.sum(np.array(img)))
    data['img_metas'][0]._data = [[data['img_metas'][0]._data]]
    data = {'img_metas':data['img_metas'],'img':[data['img'][0].unsqueeze(0)]}
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)  #{'img_metas':data['img_metas'],'img':[data['img'][0].unsqueeze(0)]}
    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)
    for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        img = mmcv.imread(img_show)
        img = img.copy()
        order_result = result[i][0][list(reversed(list(np.argsort(result[i][0][:, -1]))))]
        new_result = iou_delete(order_result)
        bbox_result, segm_result = [new_result[:3]], None #result[i],取top 3个框
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        segms = None
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=["hsil", ],
            score_thr=0.03,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            mask_color=None,
            thickness=2,
            font_size=13,
            win_name='',
            show=False,
            wait_time=0,
            out_file=None)
        #Image.fromarray(img).save("/data2/hhp/project/cervix/dual_cervix_detection/testoutanaly/visua/atss_r101_fpn_1x_iodine_hsil_totalanno_e9/det_acid_" + img_meta['ori_filename'])
        #cv2.imwrite("/data2/hhp/project/cervix/dual_cervix_detection/testoutanaly/visua/atss_r101_fpn_1x_iodine_hsil_totalanno_e9/det_iodine_"+'02790641_2019-04-28_3.jpg',img)
    return img


app = Flask(__name__,static_folder='/data2/hhp/online/cervix_detect/savemid/', static_url_path='/file')


@app.route('/cervix_detect', methods=['GET', 'POST'])
def handle_param():
    if request.method == 'POST':
        try:
            #img_url = request.form['url']
            img = request.files['file']
            pic_name = img.filename[:-3] +'png'
            path = "/data2/hhp/online/cervix_detect/savemid/" + pic_name
        except:
            result = "POST参数key name错误，示例：\n" \
                     "key: 'url'，value: 图片url"
            status = 0
        else:
            img.save(path)
            img_url = path
            try:
                ori_img = np.array(url_loader(img_url))
            except:
                result = '图像下载错误'
                status = 0
            else:
                try:
                    result = comput(ori_img,pic_name)
                    pic_root = "http://192.168.0.242:6009/file/det_iodine_" + pic_name
                    #result.save("/data/hhp/cervix_det/save/det_acid_" + pic_name)
                    cv2.imwrite("/data2/hhp/online/cervix_detect/savemid/det_iodine_" + pic_name,result)
                except:
                    result = 'inference error'
                    status = 0
                else:
                    status = 1

        return pic_root

    else:
        return 'GET not supported, use POST instead'


def build_model(cfg, model_path):
    model = get_model(cfg)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint.pop('model'))
    model.cuda()
    model.eval()

    return model


if __name__ == '__main__':
    begin_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    rank, _ = get_dist_info()
    print('Using config: ', cfg)

    # transform
    pipeline = Compose(cfg.data.test.pipeline)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = 1
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    end1 = time.time()



    #test offline
    image = Image.open('/data2/share/Cervix_Project/Images/02790641_2019-04-28_3.jpg').convert('RGB')
    #print('image array sum 1:', np.sum(np.array(image)))
    image.save("/data2/hhp/online/cervix_detect/savemid/" + '02790641_2019-04-28_3.png')
    #print(np.sum(np.array(Image.open('./iodine_ori_img.png').convert('RGB'))))
    comput(np.array(image),'02790641_2019-04-28_3.png')
    end2 = time.time()
    print('time1,time2:', end1-begin_time,end2-begin_time)

    # online use
    # app.run(
    #     host='0.0.0.0',
    #     port=6009,  # 35001
    #     debug=True
    # )


    ## test
    # img_path = '/data/lxc/Cervix/cervix_resize_600_segmentation/Images/02741656_2014-03-20_2.jpg' # or _3.jpg
    # comput(cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.uint8), name='02741656_2014-03-20_2')

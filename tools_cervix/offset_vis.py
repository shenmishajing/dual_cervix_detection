import numpy as np
import os
import json
import pickle
import mmcv
from tqdm import tqdm


def main():
    result_root_path = '/data/zhengwenhao/Result/DualCervixDetection/OffsetVisualization/faster_rcnn_dual_r50_fpn_1x_dual_zEPyVcfph'
    result_path = os.path.join(result_root_path, 'results')
    output_path = os.path.join(result_root_path, 'images')
    image_path = '../data/cervix/img'
    thickness = 3
    for file_name in tqdm(os.listdir(result_path)):
        if os.path.isfile(os.path.join(result_path, file_name)) and file_name.endswith('.pkl'):
            draw_offset_info = pickle.load(open(os.path.join(result_path, file_name), 'rb'))
            h, w = draw_offset_info['img_meta']['img_shape'][:2]
            image_shape = (1333, 800)
            acid_img_name, iodine_img_name = draw_offset_info['img_meta']['filename']
            acid_img = mmcv.imread(os.path.join(image_path, acid_img_name))
            acid_img = mmcv.imrescale(acid_img, image_shape)
            iodine_img = mmcv.imread(os.path.join(image_path, iodine_img_name))
            iodine_img = mmcv.imrescale(iodine_img, image_shape)
            for i in range(len(draw_offset_info['acid_proposals'])):
                cur_acid_img = acid_img.copy()
                cur_iodine_img = iodine_img.copy()
                acid_proposal = draw_offset_info['acid_proposals'][i, None]
                acid_gt_bboxes = draw_offset_info['acid_gt_bboxes'][i, None]
                acid_iodine_gt_bboxes = draw_offset_info['acid_iodine_gt_bboxes'][i, None]
                acid_proposal_offseted = draw_offset_info['acid_proposals_offseted'][i, None]

                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, acid_proposal, 'green', thickness = thickness, show = False)
                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, acid_gt_bboxes, 'red', thickness = thickness, show = False)
                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, acid_iodine_gt_bboxes, 'blue', thickness = thickness, show = False)
                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, acid_proposal_offseted, 'yellow', thickness = thickness, show = False)

                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, acid_proposal, 'green', thickness = thickness, show = False)
                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, acid_gt_bboxes, 'red', thickness = thickness, show = False)
                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, acid_iodine_gt_bboxes, 'blue', thickness = thickness, show = False)
                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, acid_proposal_offseted, 'yellow', thickness = thickness, show = False)

                cur_img = np.concatenate([cur_acid_img, cur_iodine_img], axis = 1)
                mmcv.imwrite(cur_img, os.path.join(output_path, os.path.splitext(acid_img_name)[0] + f'_{i}.jpg'))

            for i in range(len(draw_offset_info['iodine_proposals'])):
                cur_acid_img = acid_img.copy()
                cur_iodine_img = iodine_img.copy()
                iodine_proposal = draw_offset_info['iodine_proposals'][i, None]
                iodine_gt_bboxes = draw_offset_info['iodine_gt_bboxes'][i, None]
                iodine_acid_gt_bboxes = draw_offset_info['iodine_acid_gt_bboxes'][i, None]
                iodine_proposal_offseted = draw_offset_info['iodine_proposals_offseted'][i, None]

                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, iodine_proposal, 'green', thickness = thickness, show = False)
                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, iodine_gt_bboxes, 'blue', thickness = thickness, show = False)
                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, iodine_acid_gt_bboxes, 'red', thickness = thickness, show = False)
                cur_acid_img = mmcv.imshow_bboxes(cur_acid_img, iodine_proposal_offseted, 'yellow', thickness = thickness, show = False)

                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, iodine_proposal, 'green', thickness = thickness, show = False)
                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, iodine_gt_bboxes, 'blue', thickness = thickness, show = False)
                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, iodine_acid_gt_bboxes, 'red', thickness = thickness, show = False)
                cur_iodine_img = mmcv.imshow_bboxes(cur_iodine_img, iodine_proposal_offseted, 'yellow', thickness = thickness, show = False)

                cur_img = np.concatenate([cur_iodine_img, cur_acid_img], axis = 1)
                mmcv.imwrite(cur_img, os.path.join(output_path, os.path.splitext(iodine_img_name)[0] + f'_{i}.jpg'))


def read_json():
    ann = json.load(open('/home/zhengwenhao/Project/dual_cervix_detection/mmdetection/data/cervix/hsil_annos/train_acid.json'))
    img = [img for img in ann['images'] if img['file_name'] == '01799428_2019-09-09_2.jpg'][0]
    img_id = img['id']
    a = [a for a in ann['annotations'] if a['image_id'] == img_id][0]
    shape, scale = mmcv.rescale_size((img['width'], img['height']), (1333, 800), return_scale = True)
    bbox = [b * scale for b in a['bbox']]  # [201.33333333333331, 140.0, 280.0, 189.33333333333331]
    print(ann)


if __name__ == "__main__":
    main()
    # read_json()

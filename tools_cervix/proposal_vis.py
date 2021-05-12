import argparse
import numpy as np
import os
from pycocotools.coco import COCO
import mmcv
from tqdm import tqdm 
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_dir', type=str, help="proposal dir")
    parser.add_argument('--prim', type=str, help="acid or iodine")
    parser.add_argument('--acid_data_json', type=str, help="acid coco data file")
    parser.add_argument('--iodine_data_json', type=str, help="iodoine coco data file")
    parser.add_argument('--out_dir', type=str, help="output dir for images")
    parser.add_argument('--n_to_show', type=int, default=100, help="how many proposals to put on image")
    parser.add_argument('--p_per_img', type=int, default=4, help="how many proposals to put on image")
    parser.add_argument('--n_vis_img', type=int, default=10, help="how many images to visualize")

    return parser.parse_args()


def load_img_and_proposal_path(acid_data_json, iodine_data_json, p_dir, prim):
    img_dir="/data2/luochunhua/od/mmdetection/data/cervix/img"
    acid_coco_data = COCO(acid_data_json)
    iodine_coco_data = COCO(iodine_data_json)
    img_ids = acid_coco_data.get_img_ids()
    img_path_list = []
    for i in img_ids:
        acid_info = acid_coco_data.load_imgs([i])[0]
        iodine_info = iodine_coco_data.load_imgs([i])[0]
        img_path_list.append(
            (os.path.join(img_dir, acid_info['file_name']),
            os.path.join(img_dir, iodine_info['file_name'])
            )
        )
    prim_path_list = sorted([os.path.join(p_dir, fname) for fname in os.listdir(p_dir) if "prim" in fname])
    aux_path_list = sorted([os.path.join(p_dir, fname) for fname in os.listdir(p_dir) if "aux" in fname])
    p_path_list = []
    
    for p, a in zip(prim_path_list, aux_path_list):
        if prim == "acid":
            p_path_list.append((p, a))
        elif prim == "iodine":
            p_path_list.append((a, p))
        else:
            raise "prim must be acid or iodine"
    print(p_path_list[0])
    return img_path_list, p_path_list


def put_propoals_on_image(img_path, proposal_path, save_dir, proposal_per_img=5, n_to_show=100,  target_scale=(1333, 800)):
    acid_img_path, iodine_img_path = img_path
    acid_proposal_path, iodine_proposal_path = proposal_path

    acid_img_id = os.path.split(acid_img_path)[-1].split(".")[0]
    iodine_img_id = os.path.split(iodine_img_path)[-1].split(".")[0]

    acid_img = mmcv.imread(acid_img_path)
    iodine_img = mmcv.imread(iodine_img_path)

    acid_img, scale_factor = mmcv.imrescale(acid_img, target_scale, return_scale=True, backend='cv2')
    iodine_img, scale_factor = mmcv.imrescale(iodine_img, target_scale, return_scale=True, backend='cv2')

    acid_proposals = np.load(acid_proposal_path)['arr_0'][:n_to_show]
    iodine_proposals = np.load(iodine_proposal_path)['arr_0'][:n_to_show]
    n = acid_proposals.shape[0] // proposal_per_img
    for i in range(n):
        acid_img_ = np.copy(acid_img)
        iodine_img_ = np.copy(iodine_img)
        s = i * proposal_per_img
        e = (i + 1) * proposal_per_img
        for j in range(s, e):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            acid_img_ = mmcv.imshow_bboxes(acid_img_, acid_proposals[j:j+1], colors=color, thickness=3, show=False)
            iodine_img_ = mmcv.imshow_bboxes(iodine_img_, iodine_proposals[j:j+1], colors=color, thickness=3, show=False)
        
        img_with_box = np.concatenate([acid_img_, iodine_img_], axis=1)
        out_file = os.path.join(save_dir, acid_img_id + "_{}-{}.jpg".format(s, e))
        mmcv.imwrite(img_with_box, out_file)

def main():
    args = parse_args()
    print(args.prim)
    img_path_list, p_path_list = load_img_and_proposal_path(args.acid_data_json, args.iodine_data_json, args.p_dir, args.prim) 
    img_path_list = img_path_list[:args.n_vis_img] 
    p_path_list = p_path_list[:args.n_vis_img]

    for img_path, p_path in tqdm(zip(img_path_list, p_path_list)):
        put_propoals_on_image(img_path, p_path, args.out_dir, args.p_per_img, args.n_to_show)


if __name__ == "__main__":
    main()

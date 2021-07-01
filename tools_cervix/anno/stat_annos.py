import argparse
import base64
from mmdet.core import bbox
import shutil
import json
import os
import pickle
from mmcv import Config
from tqdm import tqdm
pjoin = os.path.join
from mmdet.datasets import build_dataset
import numpy as np 
import copy


def load_pkl_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data


def save_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def load_case_id(data_dir):
    case_id_list = [
        fname[:-7]
        for fname in os.listdir(data_dir) if fname.endswith("_2.json")
    ]

    return list(set(case_id_list))


def encode(file_path):
    with open(file_path, "rb") as f:
        base64_data = base64.b64encode(f.read())

    return str(base64_data,'utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description='stat cervix annos')
    parser.add_argument('config', help='dataset config file path')
    args = parser.parse_args()

    return args


def stat_single_dataset(data_cfg):
    ds = build_dataset(data_cfg)
    total = len(ds)

    case_id_list = []
    for i in range(len(ds)):
        img_info = ds.data_infos[i]
        ann_info = ds.get_ann_info(i)
        # print(img_info)
        # print(ann_info)
        acid_bboxes, iodine_bboxes = ann_info["bboxes"]
        acid_filename, iodine_filename = img_info["filename"]
        acid_num_bboxes = acid_bboxes.shape[0]
        iodine_num_bboxes = iodine_bboxes.shape[0]

        if not (acid_num_bboxes == 1 and iodine_num_bboxes == 1):
            case_id_list.append(acid_filename[:-5])

    return case_id_list, total


def stat_dataset(cfg, save_dir):
    train_case_id_list, train_total = stat_single_dataset(cfg.data.train)
    val_case_id_list, val_total = stat_single_dataset(cfg.data.val)
    test_case_id_list, test_total = stat_single_dataset(cfg.data.test)

    print(len(train_case_id_list), len(val_case_id_list), len(test_case_id_list), sum([len(train_case_id_list), len(val_case_id_list), len(test_case_id_list)]))
    print(train_total, val_total, test_total, sum([train_total, val_total, test_total]))


def convert_to_labelme(file_path, bboxes, labels, masks):
    file_name = os.path.split(file_path)[-1]

    labelme = {
        "Version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": file_name,
        "imageData": encode(file_path),
    }
    
    for box, label, mask in zip(bboxes.tolist(), labels.tolist(), masks):
        box_shape = {
            "label": str(label),
            "points": [box[:2], box[2:]],
            "group_id": "null",
            "shape_type": "rectangle",
            "flag": {}
        }
        labelme["shapes"].append(box_shape)

        mask_shape = {
            "label": str(1),
            "points": mask,
            "group_id": "null",
            "shape_type": "polygon",
            "flag": {}
        }
        labelme["shapes"].append(mask_shape)

    return labelme


def coco_to_labelme(cfg, output_dir):
    img_dir = cfg.data.train.img_prefix

    for t in ["train", "val", "test"]:
        data_cfg = eval("cfg.data.{}".format(t))
        ds = build_dataset(data_cfg)

        for i in tqdm(range(len(ds))):
            img_info = ds.data_infos[i]
            ann_info = ds.get_ann_info(i)
            acid_bboxes, iodine_bboxes = ann_info["bboxes"]
            acid_filename, iodine_filename = img_info["filename"]
            acid_num_bboxes = acid_bboxes.shape[0]
            iodine_num_bboxes = iodine_bboxes.shape[0]

            if not (acid_num_bboxes == 1 and iodine_num_bboxes == 1):
                acid_img_src_path = os.path.join(img_dir, acid_filename)
                iodine_img_src_path = os.path.join(img_dir, iodine_filename)
                acid_anno = convert_to_labelme(acid_img_src_path, ann_info["bboxes"][0], ann_info["labels"][0], ann_info["masks"][0])
                iodine_anno = convert_to_labelme(iodine_img_src_path, ann_info["bboxes"][1], ann_info["labels"][1], ann_info["masks"][1])
                
                acid_anno_dst_path = os.path.join(output_dir, acid_filename.replace("jpg", "json"))
                iodine_anno_dst_path = os.path.join(output_dir, iodine_filename.replace("jpg", "json"))
                save_json(acid_anno, acid_anno_dst_path)
                save_json(iodine_anno, iodine_anno_dst_path)

                acid_img_dst_path = os.path.join(output_dir, acid_filename)
                iodine_img_dst_path = os.path.join(output_dir, iodine_filename)
                shutil.copy(acid_img_src_path, acid_img_dst_path)
                shutil.copy(iodine_img_src_path, iodine_img_dst_path)


def load_labelme_anno(case_id, input_dir):
    acid_anno_path = pjoin(input_dir, case_id + "_2.json")
    iodine_anno_path = pjoin(input_dir, case_id + "_3.json")

    acid_json = load_json(acid_anno_path)
    acid_bbox = []
    acid_label = []
    for x in acid_json["shapes"]:
        if x["shape_type"] == "rectangle":
            points = x["points"]
            acid_bbox.append(points[0] + points[1])
            acid_label.append(0)
    acid_bbox = np.array(acid_bbox, dtype=np.float32)
    acid_label = np.array(acid_label, dtype=np.int64)

    iodine_json = load_json(iodine_anno_path)
    iodine_bbox = []
    iodine_label = []
    for x in iodine_json["shapes"]:
        if x["shape_type"] == "rectangle":
            points = x["points"]
            iodine_bbox.append(points[0] + points[1])
            iodine_label.append(0)
    iodine_bbox = np.array(iodine_bbox, dtype=np.float32)
    iodine_label = np.array(iodine_label, dtype=np.int64)

    return acid_bbox, acid_label, iodine_bbox, iodine_label


def convert_to_coco(annos, categories, t, output_dir):
    acid_coco = {
        "images":[],
        "annotations":[],
        "categories": categories
    }
    iodine_coco = {
        "images":[],
        "annotations":[],
        "categories": categories
    }
    anno_id = 0
    for i, (k, v) in enumerate(annos.items()):
        img_info, ann_info = v
        img_info.pop("filename")
        img_info["id"] = i
        acid_img_info = copy.deepcopy(img_info)
        acid_img_info["file_name"] = k + "_2.jpg"
        acid_coco["images"].append(acid_img_info)

        iodine_img_info = copy.deepcopy(img_info)
        iodine_img_info["file_name"] = k + "_3.jpg"
        iodine_coco["images"].append(iodine_img_info)

        acid_bbox, iodine_bbox = ann_info["bboxes"]

        for abox, ibox in zip(acid_bbox.tolist(), iodine_bbox.tolist()):
            acid_coco["annotations"].append({
                "area": (abox[2] - abox[0]) * (abox[3] - abox[1]),
                "iscrowd": 0,
                "image_id": i,
                "bbox": [abox[0], abox[1], abox[2] - abox[0], abox[3] - abox[1]],
                "category_id": 1,
                "id": anno_id
            })

            iodine_coco["annotations"].append({
                "area": (ibox[2] - ibox[0]) * (ibox[3] - ibox[1]),
                "iscrowd": 0,
                "image_id": i,
                "bbox": [ibox[0], ibox[1], ibox[2] - ibox[0], ibox[3] - ibox[1]],
                "category_id": 1,
                "id": anno_id                
            })

            anno_id += 1

    save_json(acid_coco, pjoin(output_dir, "{}_acid.json".format(t)))
    save_json(iodine_coco, pjoin(output_dir, "{}_iodine.json".format(t)))


def labelme_to_coco(cfg, input_dir, output_dir):
    """ 
    框数量为1的保留；框数量大于1的，跟labelme格式的对比，需要删除部分
    """
    case_id_list = load_case_id(input_dir)
    categories = [{"id": 0, "name": "lsil"}, {"id": 1, "name": "hsil"}]

    for t in ["train", "val", "test"]:
        data_cfg = eval("cfg.data.{}".format(t))
        ds = build_dataset(data_cfg)
        tmp = dict()
        for i in tqdm(range(len(ds))):
            img_info = ds.data_infos[i]
            ann_info = ds.get_ann_info(i)
            acid_bboxes, iodine_bboxes = ann_info["bboxes"]
            acid_filename, iodine_filename = img_info["filename"]
            acid_num_bboxes = acid_bboxes.shape[0]
            iodine_num_bboxes = iodine_bboxes.shape[0]
            
            case_id = acid_filename[:-6]
            if not (acid_num_bboxes == 1 and iodine_num_bboxes == 1):
                if not case_id in case_id_list:
                    #删除掉的案例
                    continue
                else:
                    #需要更新的案例
                    acid_bbox, acid_label, iodine_bbox, iodine_label = load_labelme_anno(case_id, input_dir)
                    anno_info = dict(
                        bboxes=[acid_bbox, iodine_bbox],
                        labels=[acid_label, iodine_label]
                    )
                    tmp[case_id] = [img_info, anno_info]
            else:
                tmp[case_id] = [img_info, ann_info]

        convert_to_coco(tmp, categories, t, output_dir)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # save_dir = "/data2/luochunhua/od/mmdetection/data/cervix/hsil_annos/more_than_one"
    # output_dir = "/data2/luochunhua/od/mmdetection/data/cervix/hsil_annos/reannos_seg"
    # output_dir = "/data2/luochunhua/od/mmdetection/data/cervix/hsil_annos/reannos"
    # stat_dataset(cfg, save_dir)
    # coco_to_labelme(cfg, output_dir)
    reannos_dir = "/data2/luochunhua/od/mmdetection/data/cervix/hsil_reannos/total"
    reannos_output_dir = "/data2/luochunhua/od/mmdetection/data/cervix/hsil_reannos/"
    labelme_to_coco(cfg, reannos_dir, reannos_output_dir)

    
# 4543 1302 647

if __name__ == "__main__":
    main()
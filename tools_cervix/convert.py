import os 
import pickle
import json
from tqdm import tqdm
import numpy as np 
pjoin = os.path.join

src_img_dir = "/data/luochunhua/cervix/cervix_det_data/img"
src_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
src_sil_train_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/train.txt"
src_sil_valid_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/valid.txt"
src_sil_test_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/test.txt"
src_sil_total_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/total.txt"

dst_single_json_dir = "/data/luochunhua/od/mmdetection/data/cervix/annos/single"
dst_dual_pkl_dir = "/data/luochunhua/od/mmdetection/data/cervix/annos/dual"


def load_pkl_anno(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl_anno(annos, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(annos, f)


def save_json_data(data, save_path):
    with open(save_path, "w") as f:
        json.dump(data, f)


def load_case_id_from_txt(txt_path):
    with open(txt_path, "r") as f:
        case_id_list = [
            line.strip() for line in f.readlines() if len(line.strip()) > 0
        ]
    case_id_list = list(set(case_id_list))
    return case_id_list


def convert_single(acid=True):
    suffix = "acid" if acid else "iodine"
    categories = [
        {
            "id": 0,
            "name": "lsil"
        },
        {
            "id": 1,
            "name": "hsil"
        }
    ]
    def convert_single_(case_id_txt_path, anno_path, anno_id_start):
        label_map = {
            1: 0, # lsil
            2: 1  # hsil
        }
        total_case_id_list = load_case_id_from_txt(src_sil_total_path)
        case_id_list = load_case_id_from_txt(case_id_txt_path)
        src_annos = load_pkl_anno(anno_path)

        coco_images = []
        coco_annotations = []
        anno_id = anno_id_start
        for case_id in tqdm(case_id_list):
            image_id = total_case_id_list.index(case_id)
            case_id += "_2" if acid else "_3"
            anno = src_annos[case_id]

            coco_images.append({
                "file_name": case_id + ".jpg",
                "height": anno["shape"][0],
                "width": anno["shape"][1],
                "id": image_id
            })    

            for x in anno["annos"]:
                if x["label"] == 3:
                    continue

                category_id = label_map[x["label"]]
                x1, y1, x2, y2 = x["bbox"]
                coco_annotations.append({
                    "segmentation": x["segm"].tolist(),
                    "area": (x2 - x1) * (y2 - y1) ,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": category_id,
                    "id": anno_id
                })
                anno_id += 1

        return coco_images, coco_annotations, anno_id

    anno_id_start = 0
    train_images, train_annotations, anno_id_start = convert_single_(src_sil_train_path, src_anno_path, anno_id_start)
    valid_images, valid_annotations, anno_id_start = convert_single_(src_sil_valid_path, src_anno_path, anno_id_start)
    test_images, test_annotations, anno_id_start = convert_single_(src_sil_test_path, src_anno_path, anno_id_start)

    for x in ["train", "valid", "test"]:
        anno = {
            "categories": categories,
            "images": eval("{}_images".format(x)),
            "annotations": eval("{}_annotations".format(x))
        }
        json_path = pjoin(dst_single_json_dir, "{}_{}.json".format(x, suffix))
        save_json_data(anno, json_path)


def convert_dual():
    
    def convert_dual_(case_id_txt_path, anno_path):
        label_map = {
            1: 0, # lsil
            2: 1  # hsil
        }
        case_id_list = load_case_id_from_txt(case_id_txt_path)
        src_annos = load_pkl_anno(anno_path)
        
        dst_annos = []
        for case_id in case_id_list:
            
            acid_anno = src_annos[case_id + "_2"]
            iodine_anno = src_annos[case_id + "_3"]

            acid_filename = case_id + "_2.jpg"
            iodine_filename = case_id + "_3.jpg"

            acid_bbox = np.array([x["bbox"] for x in acid_anno["annos"]], dtype=np.float32)
            iodine_bbox = np.array([x["bbox"] for x in iodine_anno["annos"]], dtype=np.float32)

            acid_label = np.array([x["label"] for x in acid_anno["annos"]], dtype=np.int64)
            iodine_label = np.array([x["label"] for x in iodine_anno["annos"]], dtype=np.int64)

            anno = {
                "filename": [acid_filename, iodine_filename],
                "width": acid_anno["shape"][1],
                "height":acid_anno["shape"][0],
                "ann": {
                    "bboxes": [acid_bbox, iodine_bbox],
                    "labels": [acid_label, iodine_label]
                }
            }

            dst_annos.append(anno)

        return dst_annos

    for x in ["train", "valid", "test"]:
        annos = convert_dual_(eval("src_sil_{}_path".format(x)), src_anno_path)
        annos_path = pjoin(dst_dual_pkl_dir, x + ".pkl")
        save_pkl_anno(annos, annos_path)


def check_dual():
    test_pkl_path = pjoin(dst_dual_pkl_dir, "test.pkl")
    anno = load_pkl_anno(test_pkl_path)
    print(anno[:2])


if __name__ == "__main__":
    # convert_single(acid=True)
    # convert_single(acid=False)

    # convert_dual()
    check_dual()
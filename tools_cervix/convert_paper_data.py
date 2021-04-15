import os 
import pickle
import json
from tqdm import tqdm
import numpy as np 
from sklearn.model_selection import train_test_split

pjoin = os.path.join

src_img_dir = "/data/luochunhua/cervix/cervix_det_data/img"
src_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/paper_annos.pkl"
src_sil_total_path = "/data/luochunhua/cervix/cervix_det_data/data_split/paper.txt"

dst_single_json_dir = "/data/luochunhua/od/mmdetection/data/cervix/paper_anno/annos"
dst_data_split_dir = "/data/luochunhua/od/mmdetection/data/cervix/paper_anno/data_split"


def load_pkl_anno(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl_anno(annos, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(annos, f)


def load_json_anno(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


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


def save_case_id_to_txt(case_id_list, save_path):
    case_id_list = list(case_id_list)
    with open(save_path, "w") as f:
        f.write("\n".join(case_id_list))


def split_sil_data():
    total_txt_path = src_sil_total_path
    total_annos_path = src_anno_path
    
    sil_txt_path =  pjoin(dst_data_split_dir, "total.txt")
    train_txt_path = pjoin(dst_data_split_dir, "train.txt")
    valid_txt_path = pjoin(dst_data_split_dir, "valid.txt")
    test_txt_path = pjoin(dst_data_split_dir, "test.txt")

    annos = load_pkl_anno(total_annos_path)
    total_case_id = load_case_id_from_txt(total_txt_path)

    sil_case_id = set()
    for case_id in total_case_id:
        acid_anno = annos[case_id + "_2"]
        iodine_anno = annos[case_id + "_3"]

        acid_has_sil = False
        for x in acid_anno["annos"]:
            if x["label"] == 1 or x["label"] == 2:
                acid_has_sil = True 
                break

        iodine_has_sil = False 
        for x in iodine_anno["annos"]:
            if x["label"] == 1 or x["label"] == 2:
                iodine_has_sil = True 
                break 
        
        if acid_has_sil and iodine_has_sil:
            sil_case_id.add(case_id)

    sil_case_id = list(sil_case_id)
    save_case_id_to_txt(sil_case_id, sil_txt_path)

    label = []
    for case_id in sil_case_id:

        acid_has_lsil = False
        acid_has_hsil = False
        acid_anno = annos[case_id + "_2"]
        for x in acid_anno["annos"]:
            if x["label"] == 1:
                acid_has_lsil = True 
            
            if x["label"] == 2:
                acid_has_hsil = True

        iodine_has_lsil = False 
        iodine_has_hsil = False
        iodine_anno = annos[case_id + "_3"]
        for x in iodine_anno["annos"]:
            if x["label"] == 1:
                iodine_has_lsil = True 
            if x["label"] == 2:
                iodine_has_hsil = True

        if not acid_has_sil and acid_has_hsil and not iodine_has_lsil and iodine_has_hsil:
            label.append(1)
        elif not acid_has_sil and acid_has_hsil and iodine_has_lsil and not iodine_has_hsil:
            label.append(2)
        elif not acid_has_sil and acid_has_hsil and iodine_has_lsil and iodine_has_hsil:
            label.append(3)
        elif acid_has_sil and not acid_has_hsil and not iodine_has_lsil and iodine_has_hsil:
            label.append(4)
        elif acid_has_sil and not acid_has_hsil and iodine_has_lsil and not iodine_has_hsil:
            label.append(5)
        elif acid_has_sil and not acid_has_hsil and iodine_has_lsil and iodine_has_hsil:
            label.append(6)
        elif acid_has_sil and acid_has_hsil and not iodine_has_lsil and iodine_has_hsil:
            label.append(7)
        elif acid_has_sil and acid_has_hsil and iodine_has_lsil and not iodine_has_hsil:
            label.append(8)
        elif acid_has_sil and acid_has_hsil and iodine_has_lsil and iodine_has_hsil:
            label.append(9)

    seed = 1
    train_case_id, valid_test_case_id, train_label, valid_label = train_test_split(sil_case_id, label, test_size=0.2, random_state=seed)
    valid_case_id, test_case_id, valid_label, test_label = train_test_split(valid_test_case_id, valid_label, test_size=0.33, random_state=seed)

    save_case_id_to_txt(train_case_id, train_txt_path)
    save_case_id_to_txt(valid_case_id, valid_txt_path)
    save_case_id_to_txt(test_case_id, test_txt_path)

    print("sil train", len(train_case_id))
    print("sil valid", len(valid_case_id))
    print("sil test", len(test_case_id))


def convert_single(acid=True):
    sil_txt_path =  pjoin(dst_data_split_dir, "total.txt")
    train_txt_path = pjoin(dst_data_split_dir, "train.txt")
    valid_txt_path = pjoin(dst_data_split_dir, "valid.txt")
    test_txt_path = pjoin(dst_data_split_dir, "test.txt")

    
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
        total_case_id_list = load_case_id_from_txt(sil_txt_path)
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
    train_images, train_annotations, anno_id_start = convert_single_(train_txt_path, src_anno_path, anno_id_start)
    valid_images, valid_annotations, anno_id_start = convert_single_(valid_txt_path, src_anno_path, anno_id_start)
    test_images, test_annotations, anno_id_start = convert_single_(test_txt_path, src_anno_path, anno_id_start)

    for x in ["train", "valid", "test"]:
        anno = {
            "categories": categories,
            "images": eval("{}_images".format(x)),
            "annotations": eval("{}_annotations".format(x))
        }
        json_path = pjoin(dst_single_json_dir, "{}_{}.json".format(x, suffix))
        save_json_data(anno, json_path)


def check_single():
    # 检查一下对应的acid和iodine他们的image_id是否相同
    acid_train_anno_path = "/data/luochunhua/od/mmdetection/data/cervix/annos/single/train_acid.json"
    iodine_train_anno_path = "/data/luochunhua/od/mmdetection/data/cervix/annos/single/train_iodine.json"

    acid_train_anno = load_json_anno(acid_train_anno_path)
    iodine_train_anno = load_json_anno(iodine_train_anno_path)

    acid_train_images = acid_train_anno["images"]
    iodine_train_images = iodine_train_anno["images"]
    diff_cnt = 0
    for i in range(len(acid_train_images)):
        if acid_train_images[i]["id"] != iodine_train_images[i]["id"] or acid_train_images[i]["file_name"][:-5] != iodine_train_images[i]["file_name"][:-5]:
            diff_cnt += 1
        # exit(-1)
    print(diff_cnt)
    #! 对应的acid和iodine具有相同的image_id


def check_paper_anno():
    #! paper_anno 每个都是有sil标注的，不存在空的标注
    anno_path = src_anno_path
    anno = load_pkl_anno(anno_path)

    cnt = 0
    for k, v in anno.items():
        if len(v["annos"]) == 0:
            # print(k)
            cnt += 1
    print("{}/{}".format(cnt, len(anno)))

    for k,v in anno.items():
        has_sil = False
        for x in v["annos"]:
            if x["label"] == 1 or x["label"] == 2:
                has_sil = True 
                break
        if not has_sil:
            print(k)        


if __name__ == "__main__":
    split_sil_data()
    convert_single(acid=True)
    convert_single(acid=False)

    # check_single()

    # check_paper_anno()
import os 
import pickle
import json

src_img_dir = "/data/luochunhua/cervix/cervix_det_data/img"
src_anno_dir = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
src_sil_train_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/train.txt"
src_sil_valid_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/valid.txt"
src_sil_test_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/test.txt"
src_sil_total_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/total.txt"

dst_single_train_json_path = ""
dst_single_valid_json_path = ""
dst_single_test_json_path = ""


def load_pkl_anno(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
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


def convert_single(acid=True):
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
        for case_id in case_id_list:
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
                coco_annotations.append({
                    "segmentation": x["segm"].tolist(),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": x["bbox"]
                })

            

            pass


        pass



    train_images = []
    train_annotations = []

    

    pass


if __name__ == "__main__":
    


    pass
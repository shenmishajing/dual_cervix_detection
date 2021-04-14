import os 
import json 

def load_json_anno(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json_data(data, save_path):
    with open(save_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    src_acid_anno_path = "/data/luochunhua/od/mmdetection/data/cervix/sil_annos/test_acid.json"
    src_iodine_anno_path = "/data/luochunhua/od/mmdetection/data/cervix/sil_annos/test_iodine.json"
    dst_acid_anno_path = "/data/luochunhua/od/mmdetection/data/cervix/sil_annos/debug_acid.json"
    dst_iodine_anno_path = "/data/luochunhua/od/mmdetection/data/cervix/sil_annos/debug_iodine.json"

    src_acid_anno = load_json_anno(src_acid_anno_path)
    src_iodine_anno = load_json_anno(src_iodine_anno_path)
    case_id_list = [
        "05992832_2013-10-04",
        "06427576_2016-11-21",
        "01494799_2017-11-01",
        "03980995_2015-05-05",
        '02883565_2019-06-17',
        '08508880_2017-09-07',
        '16017643_2017-04-13',
        '02303734_2018-04-10'
    ]
    image_id_list = [
        x["id"] for x in src_acid_anno["images"]
        if x["file_name"][:-6] in case_id_list
    ]
    acid_anno = {
        "categories": src_acid_anno["categories"],
        "images": [
            x for x in src_acid_anno["images"]
            if x["id"] in image_id_list 
        ],
        "annotations": [
            x for x in src_acid_anno["annotations"]
            if x["image_id"] in image_id_list
        ]
    }

    iodine_anno = {
        "categories": src_acid_anno["categories"],
        "images": [
            x for x in src_iodine_anno["images"]
            if x["id"] in image_id_list 
        ],
        "annotations": [
            x for x in src_iodine_anno["annotations"]
            if x["image_id"] in image_id_list
        ]
    }

    save_json_data(acid_anno, dst_acid_anno_path)
    save_json_data(iodine_anno, dst_iodine_anno_path)
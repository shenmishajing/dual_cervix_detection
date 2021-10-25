import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import  xml.dom.minidom


def readtxt(path):
    with open(path, "r") as f:
        txtx = f.readlines()
        lines = [line.strip('\n') for line in txtx]  # 去掉列表中每一个元素的换行符,若需要去掉非空的line,则if len(line.strip()) > 0
        #####case_id_list = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return lines


def writetxt(path, datain):
    with open(path, "w") as f:  # ”w"代表着每次运行都覆盖内容,a表示追加
        for i in datain:
            f.write(i + "\n")


def load_json_anno(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def load_pkl_anno(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_json_data(data, save_path):
    with open(save_path, "w") as f:
        json.dump(data, f)


def convert_pickle_to_cervix_mmdet(acid=True):
    #除标注外所有目录均在dual_cervix_detection该相对路径下
    #本函数根据划分好的训练、验证、测试数据集txt文件，读取pickle，转化成mmdetection可使用的coco训练和测试数据json格式。


    pjoin = os.path.join
    dst_data_split_dir ='/data2/hhp/dataset/KAIST/kaist-cvpr15/splits/'
    #src_anno_path = '/data/lxc/Cervix/detection/annos/anno.pkl'
    src_anno_path = "/data2/hhp/dataset/KAIST/kaist-cvpr15/kaist-paired/annotations/"
    dst_single_json_dir = '/data2/hhp/dataset/KAIST/kaist-cvpr15/mmdet_data/'

    sil_txt_path = pjoin(dst_data_split_dir, "all_data.txt")
    train_txt_path = pjoin(dst_data_split_dir, "trainval.txt")
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
            1: 0,  # lsil
            2: 1  # hsil
        }
        total_case_id_list = readtxt(sil_txt_path)
        case_id_list = readtxt(case_id_txt_path)
        #src_annos = load_pkl_anno(anno_path)

        coco_images = []
        coco_annotations = []
        anno_id = anno_id_start
        for case_id in tqdm(case_id_list):
            image_id = total_case_id_list.index(case_id)
            case_list = case_id.split('/')
            case_id += "_2" if acid else "_3"
            case_id_2 = case_list[0] +case_list[1] + "/lwir" if acid else "/visible" + case_list[2] +'.txt'
            anno = readtxt(os.path.join(src_anno_path,case_id_2))
            annowh = readtxt(os.path.join('/data2/hhp/dataset/KAIST/kaist-cvpr15/annotations-xml-new/', case_id_2[:-4]+'.xml'))
            dom = xml.dom.minidom.parse(annowh)
            root = dom.documentElement
            bb = root.getElementsByTagName('size')
            w,h =bb['width'],bb['height']

            # case_id_cp =  case_id[:-2]+ "_3" if acid else case_id[:-2]+ "_2" #对应的另一个模态
            # if 2 not in [x["label"] for x in anno["annos"]] or 2 not in [y["label"] for y in src_annos[case_id_cp]["annos"]]: #判断整张图中是否存在hsil,若不存在，用于提取含hsil框的数据
            #     continue


            coco_images.append({
                "file_name": case_id + ".jpg",
                "height": h,
                "width": w,
                "id": image_id
            })

            for x in anno:
                if x[0] == 3:  #默认if x["label"] == 3:   可以if x["label"] != 2:  #默认用于提取非炎症的数据，还可选择只提取hsil框
                    continue

                category_id = label_map[x[0]]
                x1, y1, w, h = x[1:5]
                coco_annotations.append({
                    "segmentation": [],  ###x["segm"].tolist()
                    "area": w * h,
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": [x1, y1, w, h],
                    "category_id": category_id,
                    "id": anno_id
                })
                anno_id += 1

        return coco_images, coco_annotations, anno_id

    anno_id_start = 0
    trainval_images, trainval_annotations, anno_id_start = convert_single_(train_txt_path, src_anno_path, anno_id_start)
    #val_images, val_annotations, anno_id_start = convert_single_(val_txt_path, src_anno_path, anno_id_start)
    test_images, test_annotations, anno_id_start = convert_single_(test_txt_path, src_anno_path, anno_id_start)

    for x in ["trainval", "test"]:
        anno = {
            "categories": categories,
            "images": eval("{}_images".format(x)),
            "annotations": eval("{}_annotations".format(x))
        }

        json_path = pjoin(dst_single_json_dir, "{}_{}.json".format(x, suffix))
        save_json_data(anno, json_path)






def all_data():
    test = '/data2/hhp/dataset/KAIST/kaist-cvpr15/splits/test.txt'
    trval = '/data2/hhp/dataset/KAIST/kaist-cvpr15/splits/trainval.txt'
    test_dat = readtxt(test)
    trval_dat = readtxt(trval)
    all_data = trval_dat + test_dat

    writetxt('/data2/hhp/dataset/KAIST/kaist-cvpr15/splits/all_data.txt', all_data)







if __name__=='__main__':
    #obtain_all_det_data()
    #split_det_data(path_all="data/cervix_project/detection/all_feasible_data.txt")
    #all_data()
    convert_pickle_to_cervix_mmdet(acid=True)
    #convert_pickle_to_cervix_mmdet(acid=False)
    #check_single()
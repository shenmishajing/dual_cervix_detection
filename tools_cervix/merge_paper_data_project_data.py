import os 
import shutil
import json 
import pickle
from collections import defaultdict, Counter
import numpy as np 
from tqdm import tqdm 
import cv2
import base64
from sklearn.model_selection import train_test_split

"""
    项目数据(2019年3月之前):
        ! 是单模态的检测模型，所以acid图像和iodine图像是不完全对应的，
        划分路径（需要重新划分）: 
            /data/luochunhua/cervix/maskrcnn-benchmark/datasets/cervix/split/pos_only/acid/train_pos.txt
            /data/luochunhua/cervix/maskrcnn-benchmark/datasets/cervix/split/pos_only/acid/valid_pos.txt
            /data/luochunhua/cervix/maskrcnn-benchmark/datasets/cervix/split/pos_only/acid/test_pos.txt
            
            /data/luochunhua/cervix/maskrcnn-benchmark/datasets/cervix/split/pos_only/iodine/train_pos.txt
            /data/luochunhua/cervix/maskrcnn-benchmark/datasets/cervix/split/pos_only/iodine/valid_pos.txt
            /data/luochunhua/cervix/maskrcnn-benchmark/datasets/cervix/split/pos_only/iodine/test_pos.txt

        图片(醋酸+碘)路径:
            /data/lxc/Cervix/cervix_resize_600_segmentation/Images

        标注(醋酸+碘)路径:
            /data/lxc/Cervix/detection/annos/anno.pkl
            {
                "08015425_2015-12-14_2":{
                    'shape': (600, 746),
                    'annos': [
                        {
                            'box': [233, 305, 379, 354],
                            'segmentation': {
                                'size': [600, 746], 
                                'counts': b'ljX4<\\b000001N1000001O0000001N1000001O00001N101O001O0O2O001O0O2O001O0O2O000O1000000O1000000O1000000O1000000O1000000O1000000O1000000O100000000O100000000O10000000O10O1000O1000O01000000O0100000O101O0O10001O0O10010O01O12N5J5L2N0O1000000O2O00000O100O1O0000001O00000000001O00000000001O0000000000001O000Jhff6'
                                }, 
                            'label': 1
                        }, 
                        {
                            'box': [329, 377, 460, 493],
                            'segmentation': {
                                'size': [600, 746],
                                'counts': b'bWQ61gb02N2N2N2N1O2N2N2N1O0000001N1O100O1O1O1O2N100O1O0F;N1O2N101N1O1O2N1O2N1O1O2N1O2N1N3N1N210N2N101N101N2O001N2O1N2N1M4M3L4N2O1O001O1O1O1O0011O1N2N2N2N2N2N2N2M4M4L3M4L4L3M4M2N001O001O001O001O1O001O001O001O001O1O001O1O001O1O001O001N2O001O1O001O1O2N2N2N2N2N1O2N2K5J6JfTW5'
                                }, 
                            'label': 2
                        }
                    ]
                }

            }


    论文数据:
        !双模态模型，acid图像和iodine图像是对应的
        划分路径:  
            /data/lxc/Cervix/paper_data/data_split/det/sil/train.txt
            /data/lxc/Cervix/paper_data/data_split/det/sil/valid.txt
            /data/lxc/Cervix/paper_data/data_split/det/sil/test.txt
        
        图像路径: 
            /data/lxc/Cervix/paper_data/imgs
        
        标注路径: 
            /data/lxc/Cervix/paper_data/annos.pkl
            annos.pkl   所有json图象的标注信息，字典格式：
            {
                '01696740_2015_09-17_3': {
                    'shape': [h, w],
                    'annos: [
                        {
                            'bbox': [x1, y1, x2, y2],
                            'segm': np.array(points),
                            'label': 1  # lsil: 1; hsil: 2; in: 3
                        },
                        {
                            'bbox': [x1, y1, x2, y2],
                            'segm': np.array(points),
                            'label': 2  # lsil: 1; hsil: 2; in: 3
                        }
                    ]
                },

                '01696740_2015_09-17_3': {
                    'shape': [h, w],
                    'annos: [
                        {
                            'bbox': [x1, y1, x2, y2],
                            'segm': np.array(points),
                            'label': 1  # lsil: 1; hsil: 2; in: 3
                        }
                    ]
                },
            }
            ! 一部分醋酸图像的标注经过处理了
            /data/ctt/data/CervixData/paper_data/new_test_annos_hsil_norm.pkl

    生成新的论文数据:
        8: 1: 1
        划分路径:
            /data/luochunhua/cervix/cervix_det_data/data_split/sil/train.txt
            /data/luochunhua/cervix/cervix_det_data/data_split/sil/valid.txt
            /data/luochunhua/cervix/cervix_det_data/data_split/sil/test.txt

            /data/luochunhua/cervix/cervix_det_data/data_split/hsil/train.txt
            /data/luochunhua/cervix/cervix_det_data/data_split/hsil/valid.txt
            /data/luochunhua/cervix/cervix_det_data/data_split/hsil/test.txt

            /data/luochunhua/cervix/cervix_det_data/data_split/paper.txt
            /data/luochunhua/cervix/cervix_det_data/data_split/project.txt

        图像路径:
            /data/luochunhua/cervix/cervix_det_data/img

        标注路径:
            /data/luochunhua/cervix/cervix_det_data/anno/total.pkl

"""

def load_case_id_from_txt(case_id_txt, data_type=None):

    with open(case_id_txt, "r") as f:
        if data_type == "acid":
            case_id_list = [
                line.strip().replace("_2", "") for line in f.readlines() if len(line.strip()) > 0
            ]
        elif data_type == "iodine":
            case_id_list = [
                line.strip().replace("_3", "") for line in f.readlines() if len(line.strip()) > 0
            ]
        else:
            case_id_list = [
                line.strip() for line in f.readlines() if len(line.strip()) > 0
            ]

    return list(set(case_id_list))


def save_case_id_to_txt(case_id_list, save_path):
    case_id_list = list(case_id_list)
    with open(save_path, "w") as f:
        f.write("\n".join(case_id_list))


def load_json_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def load_pkl_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data


def save_pkl_data(pkl_data, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)


def move_project_data():
    # anno_path = "/data/lxc/Cervix/detection/annos/anno.pkl"
    #! 把标注中分割标注改了,原来的分割标注为maskrcnn的格式
    #! 用这个脚本生成 /data/luochunhua/cervix/maskrcnn-benchmark/tools/ttt.py
    anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/project_annos_maskrcnn.pkl"
    src_img_dir = "/data/lxc/Cervix/cervix_resize_600_segmentation/Images"

    project_img_dir = "/data/luochunhua/cervix/cervix_det_data/img"
    project_case_id_txt = "/data/luochunhua/cervix/cervix_det_data/data_split/project.txt"
    project_anno_save_path = "/data/luochunhua/cervix/cervix_det_data/anno/project_annos.pkl"
    
    annos = load_pkl_data(anno_path)
    acid_set = set([k[:-2] for k in annos.keys() if k.endswith("_2")])
    iodine_set = set([k[:-2] for k in annos.keys() if k.endswith("_3")])
    inter_set = acid_set.intersection(iodine_set)

    # 碘、醋酸的图像的shape要相同
    not_empty_set = set()
    for case_id in inter_set:
        acid_anno = annos[case_id + "_2"]
        iodine_anno = annos[case_id + "_3"]
        if len(acid_anno["annos"]) and len(iodine_anno["annos"]) and acid_anno["shape"] == iodine_anno["shape"]:
            not_empty_set.add(case_id)
        

    save_case_id_to_txt(list(not_empty_set), project_case_id_txt)

    inter_annos = dict()
    for case_id in not_empty_set:
        inter_annos[case_id + "_2"] = annos[case_id + "_2"]
        inter_annos[case_id + "_3"] = annos[case_id + "_3"]
    
    new_annos = dict()
    for k, v in inter_annos.items():
        new_annos[k] = {
            "shape": list(v["shape"]),
            "annos":[
                {
                    "bbox": x["box"],
                    "segm": x["segm"], # 空标注
                    "label": x["label"]
                } #! 存在一些标注错误，如分割标注只有一、两个点
                for x in v["annos"] if len(x["segm"]) > 2
                
            ]
        }
    print(len(not_empty_set))
    save_pkl_data(new_annos, project_anno_save_path)


def move_paper_data():
    src_img_path = "/data/lxc/Cervix/paper_data/imgs"
    src_anno_dir = "/data/lxc/Cervix/paper_data/annos.pkl"

    train_txt_path = "/data/lxc/Cervix/paper_data/data_split/det/sil/train.txt"
    valid_txt_path = "/data/lxc/Cervix/paper_data/data_split/det/sil/valid.txt"
    test_txt_path = "/data/lxc/Cervix/paper_data/data_split/det/sil/test.txt"

    paper_case_id_txt = "/data/luochunhua/cervix/cervix_det_data/data_split/paper.txt"
    paper_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/paper_annos.pkl"

    train_case_id = load_case_id_from_txt(train_txt_path)
    valid_case_id = load_case_id_from_txt(valid_txt_path)
    test_case_id = load_case_id_from_txt(test_txt_path)

    paper_case_id = []
    paper_case_id.extend(train_case_id)
    paper_case_id.extend(valid_case_id)
    paper_case_id.extend(test_case_id)

    paper_anno = load_pkl_data(src_anno_dir)
    filtered_paper_case_id = []
    filtered_paper_anno = dict()
    for case_id in paper_case_id:
        acid_anno = paper_anno[case_id + "_2"]
        iodine_anno = paper_anno[case_id + "_3"]
        #! 剔除不正确的分割标注
        acid_anno["annos"] = [
            ann for ann in acid_anno["annos"] if len(ann["segm"]) > 2
        ]

        iodine_anno["annos"] = [
            ann for ann in iodine_anno["annos"] if len(ann["segm"]) > 2
        ]


        if acid_anno["shape"] == iodine_anno["shape"] and len(acid_anno["annos"]) and len(iodine_anno["annos"]):

            filtered_paper_case_id.append(case_id)
            filtered_paper_anno[case_id + "_2"] = acid_anno
            filtered_paper_anno[case_id + "_3"] = iodine_anno
    
    save_case_id_to_txt(filtered_paper_case_id, paper_case_id_txt)
    save_pkl_data(filtered_paper_anno, paper_anno_path)

    print(len(filtered_paper_case_id))


def merge():
    paper_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/paper.txt"
    paper_annos_path = "/data/luochunhua/cervix/cervix_det_data/anno/paper_annos.pkl"
    paper_image_dir = "/data/lxc/Cervix/paper_data/imgs"

    project_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/project.txt"
    project_annos_path = "/data/luochunhua/cervix/cervix_det_data/anno/project_annos.pkl"
    project_image_dir = "/data/lxc/Cervix/cervix_resize_600_segmentation/Images"

    total_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/total.txt"
    total_annos_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
    total_img_dir = "/data/luochunhua/cervix/cervix_det_data/img/"

    paper_case_id = load_case_id_from_txt(paper_txt_path)
    project_case_id = load_case_id_from_txt(project_txt_path)
    union_case_id = []
    union_case_id.extend(paper_case_id)
    union_case_id.extend(project_case_id)
    union_case_id = list(set(union_case_id))

    save_case_id_to_txt(union_case_id, total_txt_path)

    union_annos = dict()
    paper_anno = load_pkl_data(paper_annos_path)
    project_anno = load_pkl_data(project_annos_path)
    for case_id in tqdm(paper_case_id):
        union_annos[case_id + "_2"] = paper_anno[case_id + "_2"]
        union_annos[case_id + "_3"] = paper_anno[case_id + "_3"]
        src_acid_img_path = os.path.join(paper_image_dir, case_id + "_2.jpg")
        dst_acid_img_path = os.path.join(total_img_dir, case_id + "_2.jpg")
        # shutil.copy(src_acid_img_path, dst_acid_img_path)

        src_iodine_img_path = os.path.join(paper_image_dir, case_id + "_3.jpg")
        dst_iodine_img_path = os.path.join(total_img_dir, case_id + "_3.jpg")
        # shutil.copy(src_iodine_img_path, dst_iodine_img_path)


    for case_id in tqdm(set(project_case_id).difference(set(paper_case_id))):
        union_annos[case_id + "_2"] = project_anno[case_id + "_2"]
        union_annos[case_id + "_3"] = project_anno[case_id + "_3"]
        src_acid_img_path = os.path.join(project_image_dir, case_id + "_2.jpg")
        dst_acid_img_path = os.path.join(total_img_dir, case_id + "_2.jpg")
        # shutil.copy(src_acid_img_path, dst_acid_img_path)

        src_iodine_img_path = os.path.join(project_image_dir, case_id + "_3.jpg")
        dst_iodine_img_path = os.path.join(total_img_dir, case_id + "_3.jpg")
        # shutil.copy(src_iodine_img_path, dst_iodine_img_path)
    print("paper", len(paper_case_id),"project", len(set(project_case_id).difference(set(paper_case_id))))
    save_pkl_data(union_annos, total_annos_path)


def stat_paper_data():
    paper_img_path = "/data/lxc/Cervix/paper_data/imgs"
    paper_anno_dir = "/data/lxc/Cervix/paper_data/annos.pkl"
    train_txt_path = "/data/lxc/Cervix/paper_data/data_split/det/sil/train.txt"
    valid_txt_path = "/data/lxc/Cervix/paper_data/data_split/det/sil/valid.txt"
    test_txt_path = "/data/lxc/Cervix/paper_data/data_split/det/sil/test.txt"

    train_case_id = load_case_id_from_txt(train_txt_path)
    valid_case_id = load_case_id_from_txt(valid_txt_path)
    test_case_id = load_case_id_from_txt(test_txt_path)
    annos = load_pkl_data(paper_anno_dir)

    img_shape_list = [[],[]]
    anno_shape_list = [[],[]]
    for case_id_list in [train_case_id, valid_case_id, test_case_id]:
        for case_id in tqdm(case_id_list):
            for i, x in enumerate(["_2.jpg", "_3.jpg"]):
                img_name = case_id + x 
                img_path = os.path.join(paper_img_path, img_name)
                img = cv2.imread(img_path)
                img_shape = img.shape
                img_shape_list[i].append(img_shape)

                anno_shape = annos[case_id + x[:2]]["shape"]
                anno_shape_list[i].append(tuple(anno_shape))

    print(Counter(img_shape_list[0]))
    print(Counter(img_shape_list[1]))
    # Counter({(600, 900, 3): 4838, (600, 733, 3): 2181, (570, 696, 3): 1221, (600, 746, 3): 886, (600, 800, 3): 874, (600, 750, 3): 140, (600, 745, 3): 24, (600, 744, 3): 1})
    # Counter({(600, 900, 3): 4815, (600, 733, 3): 2181, (570, 696, 3): 1221, (600, 746, 3): 907, (600, 800, 3): 874, (600, 750, 3): 140, (600, 745, 3): 26, (600, 744, 3): 1})
    print(Counter(anno_shape_list[0]))
    print(Counter(anno_shape_list[1]))
    # Counter({(600, 900): 4838, (600, 733): 2181, (570, 696): 1221, (600, 746): 886, (600, 800): 874, (600, 750): 140, (600, 745): 24, (600, 744): 1})
    # Counter({(600, 900): 4815, (600, 733): 2181, (570, 696): 1221, (600, 746): 907, (600, 800): 874, (600, 750): 140, (600, 745): 26, (600, 744): 1})


def split_data():
    total_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/total.txt"
    total_annos_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
    total_img_dir = "/data/luochunhua/cervix/cervix_det_data/img/"
    
    train_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/train.txt"
    valid_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/valid.txt"
    test_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/test.txt"
    seed = 1

    total_case_id = load_case_id_from_txt(total_txt_path)
    train_case_id, valid_test_case_id = train_test_split(total_case_id, test_size=0.2, random_state=seed)
    valid_case_id, test_case_id = train_test_split(valid_test_case_id, test_size=0.5, random_state=seed)

    save_case_id_to_txt(train_case_id, train_txt_path)
    save_case_id_to_txt(valid_case_id, valid_txt_path)
    save_case_id_to_txt(test_case_id, test_txt_path)


def split_sil_data():
    total_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/total.txt"
    total_annos_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"

    sil_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/total.txt"
    train_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/train.txt"
    valid_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/valid.txt"
    test_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/test.txt"

    annos = load_pkl_data(total_annos_path)
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
    # sil train 11166
    # sil valid 1396
    # sil test 1396


def split_hsil_data():
    total_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/total.txt"
    total_annos_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"

    hsil_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/hsil/total.txt"
    train_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/hsil/train.txt"
    valid_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/hsil/valid.txt"
    test_txt_path = "/data/luochunhua/cervix/cervix_det_data/data_split/hsil/test.txt"

    annos = load_pkl_data(total_annos_path)
    total_case_id = load_case_id_from_txt(total_txt_path)

    hsil_case_id = set()
    for case_id in total_case_id:
        acid_anno = annos[case_id + "_2"]
        iodine_anno = annos[case_id + "_3"]

        acid_has_hsil = False
        for x in acid_anno["annos"]:
            if x["label"] == 2:
                acid_has_hsil = True 
                break

        iodine_has_hsil = False 
        for x in iodine_anno["annos"]:
            if x["label"] == 2:
                iodine_has_hsil = True 
                break 
        
        if acid_has_hsil and iodine_has_hsil:
            hsil_case_id.add(case_id)

    hsil_case_id = list(hsil_case_id)
    save_case_id_to_txt(hsil_case_id, hsil_txt_path)

    seed = 1
    train_case_id, valid_test_case_id  = train_test_split(hsil_case_id, test_size=0.3, random_state=seed)
    valid_case_id, test_case_id = train_test_split(valid_test_case_id, test_size=0.33, random_state=seed)

    save_case_id_to_txt(train_case_id, train_txt_path)
    save_case_id_to_txt(valid_case_id, valid_txt_path)
    save_case_id_to_txt(test_case_id, test_txt_path)
    print("hsil train", len(train_case_id))
    print("hsil valid", len(valid_case_id))
    print("hsil test", len(test_case_id))

    # hsil train 5318
    # hsil valid 665
    # hsil test 665


def show_anno_format():
    paper_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/paper_annos.pkl"
    project_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/project_annos.pkl"
    total_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"

    paper_anno = load_pkl_data(paper_anno_path)
    project_anno = load_pkl_data(project_anno_path)
    total_anno = load_pkl_data(total_anno_path)

    cnt = 0
    for k,v in paper_anno.items():
        if len(v["annos"]) == 0:
            cnt += 1
    print("{}/{}".format(cnt, len(paper_anno)))
    cnt = 0
    for k,v in project_anno.items():
        if len(v["annos"]) == 0:
            cnt += 1
    print("{}/{}".format(cnt, len(project_anno)))

    cnt = 0
    for k,v in total_anno.items():
        if len(v["annos"]) == 0:
            cnt += 1
    print("{}/{}".format(cnt, len(total_anno)))
    

def show_paper_format():
    anno_path = "/data/lxc/Cervix/paper_data/annos.pkl"
    # anno_path = "/data/lxc/Cervix/detection/annos/anno.pkl"

    anno = load_pkl_data(anno_path)

    for k,v in anno.items():
        for x in v["annos"]:
            if x["label"] == 3:
                print(k)
                break


def check_total_anno():
    anno_path = "/data/lxc/Cervix/detection/annos/anno.pkl"
    anno = load_pkl_data(anno_path)

    cnt = 0
    for k, v in anno.items():
        if len(v["annos"]) == 0:
            # print(k)
            cnt += 1
    print("{}/{}".format(cnt, len(anno)))

    for k,v in anno.items():
        print(k, v)
        break


def check_img_shape():
    total_txt_path = "/data/luochunhua/cervix/cervix_det/data_split/total.txt"
    case_id_list = []

    pass 

if __name__ == "__main__":
    move_project_data()
    move_paper_data()
    merge()
    # split_data()

    # show_anno_format()
    # show_paper_format()
    split_sil_data()
    split_hsil_data()


# paper 10122 project 3791
# sil train 11130
# sil valid 1864
# sil test 919
# hsil train 5299
# hsil valid 662
# hsil test 663

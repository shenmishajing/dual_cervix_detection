import pickle
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


def obtain_all_det_data():
    #此部分代码用以获得成对可用的双模态全部数据，image,mask名称、size都已经对应，且均存在病变标注(包括炎症标注为3)，但双模态病变标注并未对应。
    pickle_file_list = []
    file=open('/data/lxc/Cervix/detection/annos/anno.pkl',"rb")
    pickle_data=pickle.load(file)
    pickle_key = list(pickle_data.keys())
    for j in pickle_key:
        annos_data = pickle_data[j]['annos']
        if annos_data != []:
            pickle_file_list.append(j)


    path_img='/data/lxc/Cervix/cervix_resize_600_segmentation/Images/'
    path_mask='/data/lxc/Cervix/cervix_resize_600_segmentation/Masks/'


    masks = os.listdir(path_mask)
    mask_double = [k for k in masks if (k[-5:]=='2.gif' and k[:-5]+'3.gif' in masks) or (k[-5:]=='2.png' and k[:-5]+'3.png' in masks)
                    or (k[-5:]=='2.png' and k[:-5]+'3.gif' in masks) or (k[-5:]=='2.gif' and k[:-5]+'3.png' in masks)]

    det_file_all = []
    for i in mask_double:
        if i[-5:-3]=='2.':
            acid_img_path =  os.path.join(path_img,i[:-5]+'2.jpg')
            iodine_img_path = os.path.join(path_img, i[:-5] + '3.jpg')
            if os.path.exists(acid_img_path) and os.path.exists(iodine_img_path): #判断mask对应的图像是否存在
                ma_im_acid = np.array(Image.open(os.path.join(path_mask, i)))
                ma_im_iodine = np.array(Image.open(os.path.join(path_mask, i[:-5]+'3.'+i[-3:]))) #mask_double的获得中验证了双模态数据以相同的数据格式结尾
                if (ma_im_acid.shape[1],ma_im_acid.shape[0]) == Image.open(acid_img_path).size and \
                        (ma_im_iodine.shape[1],ma_im_iodine.shape[0]) == Image.open(iodine_img_path).size and \
                        ma_im_acid.shape == ma_im_iodine.shape:
                    if np.max(ma_im_acid)>0 and np.max(ma_im_iodine)>0 and i[:-5]+'2' in pickle_file_list and i[:-5]+'3' in pickle_file_list:
                        det_file_all.append(i[:-6])

    print('det_file_all:',len(det_file_all))


    with open("data/cervix_project/detection/all_feasible_data.txt","w") as f:  #”w"代表着每次运行都覆盖内容,a表示追加
        for i in det_file_all:
            f.write(i + "\n")



def readtxt(path):
    with open(path, "r") as f:
        txtx = f.readlines()
        lines = [line.strip('\n') for line in txtx]   # 去掉列表中每一个元素的换行符,若需要去掉非空的line,则if len(line.strip()) > 0
        #####case_id_list = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return  lines



def writetxt(path,datain):
    with open(path,"w") as f:  #”w"代表着每次运行都覆盖内容,a表示追加
        for i in datain:
            f.write(i + "\n")



def split_det_data(path_all="data/cervix_project/detection/all_feasible_data.txt"):
    # 此部分代码用以对获得成对可用的双模态全部数据进行数据划分，需要剔除只含炎症的数据，获得训练、验证和测试。
    #按图像size,图像标签生成数据划分的标签y，再进行分层抽样。


    file = open('/data/lxc/Cervix/detection/annos/anno.pkl', "rb")
    pickle_data = pickle.load(file)

    path_img = '/data/lxc/Cervix/cervix_resize_600_segmentation/Images/'

    data_all = readtxt(path_all)
    label_all = []
    data_all_last = []
    # k1 =0
    # k2 = 0
    for datai in data_all:
        acid_lab = pickle_data[datai+'_2']['annos']
        acid_lab = [int(labi['label']) for labi in acid_lab]
        iodine_lab = pickle_data[datai + '_3']['annos']
        iodine_lab = [int(labi['label']) for labi in iodine_lab]
        lab = list(set(acid_lab + iodine_lab))


        # #判断image读取的size是否与pickle中的size一致,k1和k2输出均为13912,因为size可以直接使用pickle中的。
        # acid_img_path = os.path.join(path_img, datai + '_2.jpg')
        # iodine_img_path = os.path.join(path_img, datai + '_3.jpg')
        # k1 +=1
        # if np.array(Image.open(acid_img_path)).shape[:-1] == pickle_data[datai+'_2']['shape'] and \
        #         np.array(Image.open(iodine_img_path)).shape[:-1] == pickle_data[datai+'_3']['shape']:
        #     k2+=1



        if len(lab)==1 and lab[-1]==3:
            continue  #2020年之前给的数据中，不存在只有炎症的数据，一般是炎症和lsil\hsil共存。
        else:
            # #lab method 1 根据acid和iodine的情况，具体划分成10类,但其中几种情况不存在数据，有几种情况数量量太少无法划分。
            # if list(set(acid_lab))==[1] and list(set(iodine_lab))==[1]:
            #     lab=0
            # elif list(set(acid_lab))==[1] and list(set(iodine_lab))==[1,2]:
            #     lab=1
            # elif list(set(acid_lab))==[1] and list(set(iodine_lab))==[2]:
            #     lab=2
            # elif list(set(acid_lab))==[2] and list(set(iodine_lab))==[1]:
            #     lab=3
            # elif list(set(acid_lab))==[2] and list(set(iodine_lab))==[1,2]:
            #     lab=4
            # elif list(set(acid_lab))==[2] and list(set(iodine_lab))==[2]:
            #     lab=5
            # elif list(set(acid_lab))==[1,2] and list(set(iodine_lab))==[1]:
            #     lab=6
            # elif list(set(acid_lab))==[1,2] and list(set(iodine_lab))==[1,2]:
            #     lab=7
            # elif list(set(acid_lab))==[1,2] and list(set(iodine_lab))==[2]:
            #     lab=8
            # elif 3 in list(set(acid_lab)) or 3 in list(set(iodine_lab)):
            #     lab = 9
            # else:
            #     print('error happen')  #不存在



            # #lab method 2 直接取最严重的病变作为该人的病种标签，用于划分数据
            lab = max(acid_lab + iodine_lab)
            size = pickle_data[datai+'_2']['shape']
            label = str(lab) +'_' +str(size)  #按lab method 1取值，有些类别到此处只有1个样本，无法划分数据。
            label_all.append(label)
            data_all_last.append(datai)


    #print(k1,k2)
    print(len(data_all),len(label_all),len(data_all_last))
    seed = 1
    X_train, X_valtest, y_train, y_valtest = train_test_split(data_all_last, label_all, test_size=0.2, stratify=label_all, random_state=seed)  # 按y比例分层抽样，通过用于分类问题
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, stratify=y_valtest, random_state=seed)  # 按y比例分层抽样，通过用于分类问题

    writetxt("data/cervix_project/detection/train.txt", X_train)
    writetxt("data/cervix_project/detection/val.txt", X_val)
    writetxt("data/cervix_project/detection/test.txt", X_test)


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
    dst_data_split_dir ='data/cervix_project/detection/'
    #src_anno_path = '/data/lxc/Cervix/detection/annos/anno.pkl'
    src_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
    dst_single_json_dir = 'data/cervix_project/detection/annos_frompaper/'

    sil_txt_path = pjoin(dst_data_split_dir, "all_feasible_data.txt")
    train_txt_path = pjoin(dst_data_split_dir, "train.txt")
    val_txt_path = pjoin(dst_data_split_dir, "val.txt")
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
                    "segmentation": [],  ###x["segm"].tolist()
                    "area": (x2 - x1) * (y2 - y1),
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
    val_images, val_annotations, anno_id_start = convert_single_(val_txt_path, src_anno_path, anno_id_start)
    test_images, test_annotations, anno_id_start = convert_single_(test_txt_path, src_anno_path, anno_id_start)

    for x in ["train", "val", "test"]:
        anno = {
            "categories": categories,
            "images": eval("{}_images".format(x)),
            "annotations": eval("{}_annotations".format(x))
        }

        json_path = pjoin(dst_single_json_dir, "{}_{}.json".format(x, suffix))
        save_json_data(anno, json_path)



def check_single():
    # 检查一下对应的acid和iodine他们的image_id是否相同
    acid_train_anno_path = "/data2/hhp/project/cervix/dual_cervix_detection/data/cervix_project/detection/json/train_acid.json"
    iodine_train_anno_path = "/data2/hhp/project/cervix/dual_cervix_detection/data/cervix_project/detection/json/train_iodine.json"

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






if __name__=='__main__':
    #obtain_all_det_data()
    #split_det_data(path_all="data/cervix_project/detection/all_feasible_data.txt")
    #convert_pickle_to_cervix_mmdet(acid=True)
    #convert_pickle_to_cervix_mmdet(acid=False)
    check_single()


























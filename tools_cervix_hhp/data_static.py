import pickle
import os
import cv2
import numpy as np
from PIL import Image

# #original all image
# path_img='/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/img'
# imgs = os.listdir(path_img)
# lena = len([i for i in imgs])
# lenb = len([i for i in imgs if i[-5:]=='3.jpg'])
# lenc = len([i for i in imgs if i[-5:]=='2.jpg'])
# leng = len([i for i in imgs if i[-5:]=='3.jpg' and i[:-5]+'2.jpg' in imgs])
# size_img = []
# for im in imgs:
#     pa_im = os.path.join(path_img,im)
#     shap = Image.open(pa_im).size
#     if shap not in size_img:
#         size_img.append(shap)
# print(size_img)
#
# #original all mask
# path_mask='/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/mask'
# masks = os.listdir(path_mask)
# lenmask_a = len([i for i in masks])
# lenmaskb = len([i for i in masks if i[-5:]=='3.gif' or i[-5:]=='3.png'])
# lenmaskc = len([i for i in masks if i[-5:]=='2.gif' or i[-5:]=='2.png'])
# lenmaskg = len([i for i in masks if (i[-5:]=='3.gif' and i[:-5]+'2.gif' in masks) or (i[-5:]=='3.png' and i[:-5]+'2.png' in masks)
#                 or (i[-5:]=='3.png' and i[:-5]+'2.gif' in masks) or (i[-5:]=='3.gif' and i[:-5]+'2.png' in masks)])
# for i in masks:
#     ma_im = np.array(Image.open(os.path.join(path_mask,i)))
#     print(np.max(ma_im),np.min(ma_im))
#
#
# #读取pickle,查看分类、检测、分割标注
# file=open('/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/annos/anno.pkl',"rb")
# data=pickle.load(file)
#
#
#
# train_iodine_txt = '/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/split/iodine/train_pos.txt'
# val_iodine_txt = '/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/split/iodine/valid_pos.txt'
# test_iodine_txt = '/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/split/iodine/test_pos.txt'
# surface_iodine_file = '/data/lxc/code/github/maskrcnn-benchmark/datasets/cervix/surface/result_iodine.pkl'







import json
data_root = 'data/cervix/'
train_acid_ann_file = '/data2/hhp/project/cervix/dual_cervix_detection/data/cervix/hsil_rereannos/train_acid.json'
train_iodine_ann_file = '/data2/hhp/project/cervix/dual_cervix_detection/data/cervix/hsil_rereannos/train_iodine.json'

val_acid_ann_file = '/data2/hhp/project/cervix/dual_cervix_detection/data/cervix/hsil_rereannos/val_acid.json'
val_iodine_ann_file = '/data2/hhp/project/cervix/dual_cervix_detection/data/cervix/hsil_rereannos/val_iodine.json'

test_acid_ann_file = '/data2/hhp/project/cervix/dual_cervix_detection/data/cervix/hsil_rereannos/test_acid.json'
test_iodine_ann_file = '/data2/hhp/project/cervix/dual_cervix_detection/data/cervix/hsil_rereannos/test_iodine.json'

def lenn(train_acid_ann_file):
    with open(train_acid_ann_file, 'r') as f:
        data = json.load(f)
    if max([i['category_id'] for i in data['annotations']])==1 and min([i['category_id'] for i in data['annotations']])==1:
        print(train_acid_ann_file,len(data['annotations']))  #category_id=1 represent hsil


lenn(train_acid_ann_file)
lenn(train_iodine_ann_file)
lenn(val_acid_ann_file)
lenn(val_iodine_ann_file)
lenn(test_acid_ann_file)
lenn(test_iodine_ann_file)





















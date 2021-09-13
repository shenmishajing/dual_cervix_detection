import pickle
import os
import cv2
import numpy as np
from PIL import Image

#original all image
path_img='/data/lxc/Cervix/cervix_resize_600_segmentation/Images/'
imgs = os.listdir(path_img)
lena = len([i for i in imgs])
lenb = len([i for i in imgs if i[-5:]=='3.jpg'])
lenc = len([i for i in imgs if i[-5:]=='2.jpg'])
leng = len([i for i in imgs if i[-5:]=='3.jpg' and i[:-5]+'2.jpg' in imgs])
# size_img = {}
# for im in imgs:
#     pa_im = os.path.join(path_img,im)
#     shap = Image.open(pa_im).size
#     if str(shap) not in size_img.keys():
#         size_img[str(shap)] = 1
#     else:
#         size_img[str(shap)] +=1
# print('size_img:',size_img)




#original all mask
path_mask='/data/lxc/Cervix/cervix_resize_600_segmentation/Masks/'
masks = os.listdir(path_mask)
lenmask_a = len([i for i in masks])
lenmaskb = len([i for i in masks if i[-5:]=='3.gif' or i[-5:]=='3.png'])
lenmaskc = len([i for i in masks if i[-5:]=='2.gif' or i[-5:]=='2.png'])
lenmaskg = len([i for i in masks if (i[-5:]=='3.gif' and i[:-5]+'2.gif' in masks) or (i[-5:]=='3.png' and i[:-5]+'2.png' in masks)
                or (i[-5:]=='3.png' and i[:-5]+'2.gif' in masks) or (i[-5:]=='3.gif' and i[:-5]+'2.png' in masks)])
masks_can = []
quzhi = []
size_mask = {}
img_exist = 0
img_exist_size = 0
for i in masks:
    ma_im = np.array(Image.open(os.path.join(path_mask,i)))
    # if os.path.exists(os.path.join(path_img,i[:-3]+'jpg')): #判断mask对应的图像是否存在
    #     img_exist +=1
    #     if (ma_im.shape[1],ma_im.shape[0]) == Image.open(os.path.join(path_img,i[:-3]+'jpg')).size:
    #         img_exist_size +=1
    #     else:
    #         print('图像和标注含不同size：', i, (ma_im.shape[1],ma_im.shape[0]), Image.open(os.path.join(path_img,i[:-3]+'jpg')).size,np.unique(ma_im))

    if np.max(ma_im)>0:
        masks_can.append(i)

    # if list(np.unique(ma_im)) not in quzhi:
    #     quzhi.append(list(np.unique(ma_im)))

    # shap = ma_im.shape
    # if str(shap) not in size_mask.keys():
    #     size_mask[str(shap)] = 1
    # else:
    #     size_mask[str(shap)] +=1

# print('mask对应的图像存在的个数：',img_exist)
# print('mask对应的图像存在且尺寸一致的个数：', img_exist_size)
# print('size_mask:',size_mask)
print('病变mask总数：', len(masks_can))
#print('mask 取值：', quzhi)




def readtxt(path):
    with open(path, "r") as f:
        txtx = f.readlines()
        lines = [line.strip('\n') for line in txtx]   # 去掉列表中每一个元素的换行符
    return  lines



train_iodine_txt = '/data/lxc/Cervix/segmentation/cervix_resize_600_segmentation/data_split/iodine/train_pos.txt'
train_iodine = readtxt(train_iodine_txt)
masks_la = [i[:-4] for i in masks]
masks_can_la = [i[:-4] for i in masks_can]
train_from = [i for i in train_iodine if i in masks_la] #与mask文件名对应数量
train_from_can = [i for i in train_iodine if i in masks_can_la] #与病变mask文件名对应量

val_iodine_txt = '/data/lxc/Cervix/segmentation/cervix_resize_600_segmentation/data_split/iodine/valid_pos.txt'
val_iodine = readtxt(val_iodine_txt)
val_from = [i for i in val_iodine if i in masks_la] #与mask文件名对应数量
val_from_can = [i for i in val_iodine if i in masks_can_la] #与病变mask文件名对应量

test_iodine_txt = '/data/lxc/Cervix/segmentation/cervix_resize_600_segmentation/data_split/iodine/test_pos.txt'
test_iodine = readtxt(test_iodine_txt)
test_from = [i for i in test_iodine if i in masks_la] #与mask文件名对应数量
test_from_can = [i for i in test_iodine if i in masks_can_la] #与病变mask文件名对应量
print('train:',len(train_iodine), len(train_from), len(train_from_can))
print('val:',len(val_iodine), len(val_from), len(val_from_can))
print('test:',len(test_iodine), len(test_from), len(test_from_can))


surface_iodine_file = '/data/lxc/Cervix/classification/surface/result_iodine.pkl'
surface_file=open(surface_iodine_file,"rb")
surface_data=pickle.load(surface_file)
print(len(surface_data))


train_acid_txt = '/data/lxc/Cervix/segmentation/cervix_resize_600_segmentation/data_split/acid/train_pos.txt'
val_acid_txt = '/data/lxc/Cervix/segmentation/cervix_resize_600_segmentation/data_split/acid/valid_pos.txt'
test_acid_txt = '/data/lxc/Cervix/segmentation/cervix_resize_600_segmentation/data_split/acid/test_pos.txt'

surface_acid_file = '/data/lxc/Cervix/classification/surface/result_acid.pkl'  #29399个样本







#查看json文件情况
path_json='/data/lxc/Cervix/cervix_resize_600_segmentation/Jsons/'
masks_la = [i[:-4] for i in masks]
masks_can_la = [i[:-4] for i in masks_can]
json_file = os.listdir(path_json)
json_seg = [i for i in json_file if i[:-5] in masks_la] #与mask文件名对应数量
json_can_seg = [i for i in json_file if i[:-5] in masks_can_la] #与病变mask文件名对应量






#读取pickle,查看分类、检测、分割标注
file=open('/data/lxc/Cervix/detection/annos/anno.pkl',"rb")
data=pickle.load(file)
key = list(data.keys())
print('all pickle:',len(key))
num_nolabel = 0
num_no2 = 0
size_pick = {}
det1 = 0
det2 = 0
label_all = {}
masks_la = [i[:-4] for i in masks]
from_seg = 0
for i in key:
    if i in masks_la:
        from_seg+=1
    size_data = data[i]['shape']

    if str(size_data) not in size_pick.keys():
        size_pick[str(size_data)] = 1
    else:
        size_pick[str(size_data)] +=1

    annos_data = data[i]['annos']
    if annos_data == []:
        num_nolabel += 1
    else:
        for lab in annos_data:
            det_bbox = lab['box']
            if det_bbox[2]>det_bbox[0] and det_bbox[3]>det_bbox[1]:
                det1+=1 #检测框标注格式为【x1,y1,x2,y2】
            if det_bbox[0] + det_bbox[2]<size_data[1] and det_bbox[1] + det_bbox[3]<size_data[0]:
                det2 +=1 #检测框标注格式为【x1,y1,w,h】
            det_seg = lab['segmentation']
            det_lab = lab['label']
            if str(det_lab) not in label_all.keys():
                label_all[str(det_lab)] = 1
            else:
                label_all[str(det_lab)] += 1


        if det_bbox ==[] or det_seg ==[] or det_lab ==[]:
            num_no2 +=1


print('num_nolabel:', num_nolabel,num_no2)
print('pickle size:', size_pick)
print('det label',label_all)
print('det style:',det1,det2)
print('label to seg:',from_seg)

































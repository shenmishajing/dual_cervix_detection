import pickle
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import mmcv


def iou(box1,box2):
    #assert box1.size()==4 and box2.size()==4,"bounding box coordinate size must be 4"
    bxmin = max(box1[0],box2[0])
    bymin = max(box1[1],box2[1])
    bxmax = min(box1[2],box2[2])
    bymax = min(box1[3],box2[3])
    bwidth = bxmax-bxmin
    bhight = bymax-bymin
    if bxmin >= bxmax or bymin >= bymax:
        print("没有重合区域")
        return 0
    else:
        inter = bwidth*bhight
        union = (box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1])-inter
        return inter/union


def trans(box1,box2):
    # b1=[box1[0]-box1[2]/2,box1[1]-box1[3]/2,box1[0]+box1[2]/2,box1[1]+box1[3]/2]
    # b2=[box2[0]-box2[2]/2,box2[1]-box2[3]/2,box2[0]+box2[2]/2,box2[1]+box2[3]/2]
    b1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    b2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    return b1,b2







def compare_pkl():
    #因前期数据中存在多个pickle文件，且不同文件训练效果差异大，因此需要对比标注的差异性。
    img_oripath='/data/lxc/Cervix/cervix_resize_600_segmentation/Images/'
    img_out_path = 'project_result/datavisua/'

    pklfile_proj=open('/data/lxc/Cervix/detection/annos/anno.pkl',"rb")
    pklfile_paper = open("/data/luochunhua/cervix/cervix_det_data/anno/total.pkl", "rb")  #total.pkl，paper_annos.pkl
    pickle_proj_data=pickle.load(pklfile_proj)
    pickle_proj_key = list(pickle_proj_data.keys())
    pickle_paper_data = pickle.load(pklfile_paper)
    pickle_paper_key = list(pickle_paper_data.keys())

    # txt_path = 'data/cervix_project/detection/'   #此段代码确认划分的数据均能在2个pickle标注文件中找到。
    # from tools_cervix_hhp.obtain_det_data import readtxt
    # lost = 0
    # lost2 =0
    # for da in ['train.txt','val.txt','test.txt']:
    #     path = os.path.join(txt_path,da)
    #     da_list = readtxt(path)
    #     for kk in da_list:
    #         if kk+'_2' not in pickle_paper_key or kk+'_3' not in pickle_paper_key:
    #             lost +=1
    #         if kk+'_2' not in pickle_proj_key or kk+'_3' not in pickle_proj_key:
    #             lost2 +=1
    # print(lost,lost2)


    # num7 = 0
    # num8 = 0
    # for j in pickle_paper_key:  #判断是否paper_ano中是否有图像不存在于proj中
    #     num7 +=1
    #     if j in pickle_proj_key:
    #         num8 +=1
    # print('len of pickle_paper_key and number in pickle_proj_key',num7,num8) #27826,27826



    # #测试total.pkl中的box是否是【x1,y1,x2,y2】格式,确认
    # det0 =0
    # det1 = 0
    # det2 = 0
    # for j in pickle_paper_key:
    #     size_data = pickle_paper_data[j]['shape']
    #     annos_pap = pickle_paper_data[j]['annos']
    #     for lab in annos_pap:
    #         det0 += 1
    #         det_bbox = lab['bbox']
    #         if det_bbox[2]>det_bbox[0] and det_bbox[3]>det_bbox[1]:
    #             det1+=1 #检测框标注格式为【x1,y1,x2,y2】
    #         if det_bbox[0] + det_bbox[2]<size_data[1] and det_bbox[1] + det_bbox[3]<size_data[0]:
    #             det2 +=1 #检测框标注格式为【x1,y1,w,h】
    # print(det0,det1,det2) #54531,54531,14438



    numall = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0
    num6 = 0
    num9 = 0
    num10 = 0
    num11 = 0
    sumiou = 0
    sumioun1 = 0
    num_shap = 0
    for j in pickle_proj_key:
        numall +=1
        if j in pickle_paper_key:
            num2 +=1
            annos_proj_data = pickle_proj_data[j]['annos']
            annos_paper_data = pickle_paper_data[j]['annos']
            annos_proj_shape = pickle_proj_data[j]['shape']
            annos_paper_shape = pickle_paper_data[j]['shape']

            if annos_proj_shape[0]==annos_paper_shape[0] and annos_proj_shape[1]==annos_paper_shape[1]:
                num_shap += 1

            if annos_proj_data != [] and annos_paper_data !=[]:
                num3 += 1

                bboxs_proj = sorted(dict(zip([i for i in range(len(annos_proj_data))], [i['box'] for i in annos_proj_data])).items(),key=lambda x: x[1][0])
                bboxs_proj = [i[1] for i in bboxs_proj]
                bboxs_paper = sorted(dict(zip([i for i in range(len(annos_paper_data))], [i['bbox'] for i in annos_paper_data])).items(),key=lambda x: x[1][0])
                bboxs_paper = [i[1] for i in bboxs_paper]


                img_path = os.path.join(img_oripath,j+'.jpg')
                img = mmcv.imread(img_path)
                img_ = np.copy(img)


                if len(bboxs_proj)==len(bboxs_paper):
                    num4 += 1
                else:
                    num5 += 1
                change_im = 0
                for id in range(min(len(bboxs_proj),len(bboxs_paper))):

                    num6 += 1
                    #a = trans(bboxs_proj[id], bboxs_paper[id])
                    a = (bboxs_proj[id], bboxs_paper[id])
                    IOU = iou(a[0], a[1])
                    sumiou += IOU
                    if int(IOU) !=1: #int(IOU) !=1:    ;  IOU <0.9;;
                        change_im += 1
                        num11 +=1
                        sumioun1 += IOU

                        #img_ = mmcv.imshow_bboxes(img_, np.array([np.array(bboxs_proj[id])]), colors=['green'], thickness=3,show=False)
                        #img_ = mmcv.imshow_bboxes(img_, np.array([np.array(bboxs_paper[id])]), colors=['red'], thickness=3, show=False)
                #if change_im>0:
                    #mmcv.imwrite(img_, os.path.join(img_out_path, j + '.jpg'))




        else:
            annos_proj_data = pickle_proj_data[j]['annos']
            if annos_proj_data == []:
                num9 +=1
            else:
                num10 += len(annos_proj_data)


    print('all img in pkl proj:',numall) #45246
    print('all img in pkl proj and paper:', num2) #27826
    print('all img in pkl proj and paper and have annos:', num3) #27826
    print('all img in pkl proj and paper and have same len annos:', num4) #27369
    print('all img in pkl proj and paper and have  not same len annos:', num5) #457
    print('all img in pkl proj and paper and have annos and have least bbox:', num6, num11) #54334---1:41315---0.9:1409
    print('average iou:',sumiou/num6) #260多个匹配框区域完全不重合 #0.97
    print('average iou:', sumioun1 / num11) #标注变过的框的平均IOU  #1:0.96---0.9:0.6275

    print('annos_proj_data not in paper annos for none and leng', num9, num10) #16522 2110

    print('number of same img have same shape:',num_shap) #27826







if __name__=='__main__':
    compare_pkl()
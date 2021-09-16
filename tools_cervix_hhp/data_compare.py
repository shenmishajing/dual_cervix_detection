import pickle
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json










def compare_pkl():
    #此部分代码用以获得成对可用的双模态全部数据，image,mask名称、size都已经对应，且均存在病变标注(包括炎症标注为3)，但双模态病变标注并未对应。






    pickle_file_list = []
    pklfile_proj=open('/data/lxc/Cervix/detection/annos/anno.pkl',"rb")
    pklfile_paper = open("/data/luochunhua/cervix/cervix_det_data/anno/paper_annos.pkl", "rb")  #total.pkl
    pickle_proj_data=pickle.load(pklfile_proj)
    pickle_proj_key = list(pickle_proj_data.keys())
    pickle_paper_data = pickle.load(pklfile_paper)
    pickle_paper_key = list(pickle_paper_data.keys())
    numall = 0
    num2 = 0
    for j in pickle_proj_key:
        numall +=1
        if j in pickle_paper_key:
            num2 +=1
            annos_proj_data = pickle_proj_data[j]['annos']
            annos_paper_data = pickle_paper_data[j]['annos']
            if annos_proj_data != [] and annos_paper_data !=[]:
                pickle_file_list.append(j)




if __name__=='__main__':
    compare_pkl()
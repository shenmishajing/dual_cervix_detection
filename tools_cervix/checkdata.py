import os 
import pickle 
import json 
from tqdm import tqdm
pjoin = os.path.join

src_img_dir = "/data/luochunhua/cervix/cervix_det_data/img"
src_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
src_sil_train_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/train.txt"
src_sil_valid_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/valid.txt"
src_sil_test_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/test.txt"
src_sil_total_path = "/data/luochunhua/cervix/cervix_det_data/data_split/sil/total.txt"


def load_case_id_from_txt(case_id_path):
    with open(case_id_path, "r") as f:
        case_id_set = set([
            line.strip() for line in f.readlines() if len(line.strip()) > 0
        ])
    return case_id_set


def load_pkl_anno(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def check():
    # 检查是不是都是由sil的标签
    def has_sil(anno):
        if len(anno) == 0:
            return False
        flag = False
        for x in anno:
            if x["label"] == 1 or x["label"] == 2:
                flag = True 
                break
        return flag

    total_annos = load_pkl_anno(src_anno_path)
    cnt_no_sil = 0
    for k, v in tqdm(total_annos.items()):
        if not has_sil(v["annos"]):
            cnt_no_sil += 1
            print(k)
    print(cnt_no_sil)


if __name__ == "__main__":
    check()
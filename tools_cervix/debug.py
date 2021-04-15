import json 
import pickle


def load_pkl_anno(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl_anno(annos, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(annos, f)


def load_json_anno(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
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


if __name__ == "__main__":

    p = "/data/luochunhua/cervix/cervix_det_data/anno/total.pkl"
    # anno = load_pkl_anno(p)
    # cnt = 0
    # for ann in anno.values():
    #     for a in ann["annos"]:
    #         if len(a["segm"]) < 2:
    #             cnt += 1
    #             break

    # print(cnt)

    # pass

    def test(**kwargs):
        print(kwargs)
        keys = kwargs.keys()
        value = kwargs.values()
        print(keys)
        print(value)


    test(a=1,b=2,c=3,d=4)
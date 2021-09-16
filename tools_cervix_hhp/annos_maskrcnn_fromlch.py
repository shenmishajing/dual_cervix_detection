import os
import pickle
from tqdm import tqdm
import cv2
from .segmentation_mask import BinaryMaskList
'''from /data/luochunhua/cervix/maskrcnn-benchmark/tools/ttt.py to transform mask label'''

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


def load_pkl_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data


def save_pkl_data(pkl_data, pkl_path):
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)


def modify_anno():
    img_dir = "/data/lxc/Cervix/cervix_resize_600_segmentation/Images"
    project_anno_path = "/data/lxc/Cervix/detection/annos/anno.pkl"
    new_project_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/project_annos_maskrcnn.pkl"
    project_annos = load_pkl_data(project_anno_path)
    new_project_annos = dict()

    for k, annos in tqdm(project_annos.items()):
        new_project_annos[k] = {
            "shape": annos["shape"],
            "annos": []
        }

        mask_list = [x["segmentation"] for x in annos["annos"]]
        size = (annos["shape"][1], annos["shape"][0])
        bml = BinaryMaskList(mask_list, size)
        masks = bml.masks.detach().numpy()
        contours = []
        for mask in masks:
            contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            contours.append(contour[0][:, 0, :])
            # print(contour[0].shape)

        for i, anno in enumerate(annos["annos"]):
            new_project_annos[k]["annos"].append({
                "box": anno["box"],
                "segm": contours[i],
                "label": anno["label"]
            })

        # print(annos)
        # print(new_project_annos[k])
        # exit(-1)
        # img_path = os.path.join(img_dir, k + ".jpg")
        # img = cv2.imread(img_path)
        # for anno in new_project_annos[k]["annos"]:
        #     img = cv2.polylines(img, [anno["segm"]], 1, (255,0,0))
        #     box = anno["box"]
        #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),(255,0,0), 1)
        #     print(anno)
        # cv2.imwrite("./1.jpg",img)
        # exit(1)
    save_pkl_data(new_project_annos, new_project_anno_path)


def vis(img, anno):
    for a in anno["annos"]:
        mask_points = a["segm"]
        box = a["box"]
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
        img = cv2.polylines(img, [mask_points], 1, (255, 0, 0))

    return img


def test():
    new_project_anno_path = "/data/luochunhua/cervix/cervix_det_data/anno/project_annos_maskrcnn.pkl"
    img_dir = "/data/lxc/Cervix/cervix_resize_600_segmentation/Images"

    annos = load_pkl_data(new_project_anno_path)

    n = 10
    for k, v in annos.items():
        img_path = os.path.join(img_dir, k + ".jpg")
        img = cv2.imread(img_path)
        img = vis(img, v)
        cv2.imwrite("./{}.jpg".format(n), img)
        n -= 1
        if n < 0:
            break

        pass


if __name__ == "__main__":
    modify_anno()
    test()
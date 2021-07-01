import json 
import os 


def save_json(json_data, save_path):
    with open(save_path, "w") as f:
        json.dump(json_data, f)


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def remove_mask(src_path, dst_path):
    annos = load_json(src_path)
    annotations = annos["annotations"]
    
    for i in range(len(annotations)):
        if "segmentation" in annotations[i]:
            annotations[i].pop("segmentation")
    annos["annotations"] = annotations
    
    save_json(annos, dst_path)


if __name__ == "__main__":
    src_dir = "data/cervix/hsil_reannos"
    dst_dir = "data/cervix/hsil_reannos_nomask"

    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        remove_mask(src_path, dst_path)

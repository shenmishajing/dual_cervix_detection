import os 
import json
import pandas as pd 
import numpy as np 
from collections import defaultdict


def parse_log(exp_dir):
    
    log_fname = [fname for fname in sorted(os.listdir(exp_dir)) if fname.endswith("json")][-1]
    log_path = os.path.join(exp_dir, log_fname)

    log_dict = defaultdict(list)
    with open(log_path, "r") as f:
        for line in f.readlines():
            log = json.loads(line.strip())
            if 'epoch' not in log or 'val' != log['mode']:
                continue

            log_dict["epoch"] = log["epoch"]
            for k in log.keys():
                if "acid" in k or "iodine" in k:
                    log_dict[k].append(log[k])
    
    max_log_dict = dict()
    max_idx_dict = dict()
    max_value_dict = dict()
    for k ,v in log_dict.items():
        if k != "epoch":
            idx = np.argmax(v)
            max_log_dict[k] = (v[idx], idx)
            max_idx_dict[k] = idx
            max_value_dict[k] = v[idx]
    return max_log_dict, max_value_dict, max_idx_dict


def main(work_dirs, save_dir):

    table_log_dict = dict()
    table_value_dict = dict()
    table_idx_dict = dict()

    for fname in sorted(os.listdir(work_dirs)):
        if fname.endswith("hsil"):
            exp_dir = os.path.join(work_dirs, fname)
            max_log_dict,  max_value_dict, max_idx_dict = parse_log(exp_dir)

            table_log_dict[fname] = max_log_dict
            table_value_dict[fname] = max_value_dict
            table_idx_dict[fname] = max_idx_dict
    df_log = pd.DataFrame.from_dict(table_log_dict, orient="index")
    df_value = pd.DataFrame.from_dict(table_value_dict, orient="index")
    df_idx = pd.DataFrame.from_dict(table_idx_dict, orient="index")

    df_log.to_csv(save_dir + "/log.csv")
    df_value.to_csv(save_dir + "/value.csv")
    df_idx.to_csv(save_dir + "/idx.csv")


if __name__ == "__main__":
    work_dirs = "/data2/luochunhua/od/mmdetection/work_dirs"
    save_dir = "/data2/luochunhua/od/mmdetection/stat"
    main(work_dirs, save_dir)
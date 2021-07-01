import os 
import shutil
from mmdet.datasets import build_dataset
from mmcv import Config
import argparse
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser(description='stat cervix annos')
    parser.add_argument('config', help='dataset config file path')
    args = parser.parse_args()

    return args


def move(cfg, src_dir, dst_dir):
    
    for t in ["train", "val", "test"]:
        data_cfg = eval("cfg.data.{}".format(t))
        ds = build_dataset(data_cfg)
        print(len(ds))
        for i in tqdm(range(len(ds))):
            img_info = ds.data_infos[i]
            acid_filename, iodine_filename = img_info["filename"]

            acid_src_path = os.path.join(src_dir, acid_filename)
            acid_dst_path = os.path.join(dst_dir, acid_filename)

            iodine_src_path = os.path.join(src_dir, iodine_filename)
            iodine_dst_path = os.path.join(dst_dir, iodine_filename)
        
            shutil.copy(acid_src_path, acid_dst_path)
            shutil.copy(iodine_src_path, iodine_dst_path)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    src_dir = "/data/luochunhua/cervix/cervix_det_data/img"
    dst_dir = "/data2/luochunhua/od/detection_datasets/cervix/img"
    move(cfg, src_dir, dst_dir)


if __name__ == "__main__":
    main()
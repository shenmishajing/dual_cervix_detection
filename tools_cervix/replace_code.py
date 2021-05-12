import re 
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, help="")
    return parser.parse_args()


def main():
    args = parse_args()
    code_py = "/data2/luochunhua/od/mmdetection/mmdet/models/roi_heads/dual_cervix_roi_head.py"
    
    with open(code_py, "r") as f:
        code = f.read()
    
    target_str = re.findall('work_dirs/(.*)/proposals', code)[0]
    new_code = code.replace(target_str, args.exp_name)

    with open(code_py, "w") as f:
        f.write(new_code)
        f.flush()


if __name__ == "__main__":
    main()
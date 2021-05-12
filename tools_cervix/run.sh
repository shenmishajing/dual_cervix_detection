#!每次必改
GPU_ID=1
NAME=dual_dualdet_faster_debug_acid_hsil

#*不需要经常变动
CONFIG_DIR=/data/luochunhua/od/mmdetection/configs/_cervix/hsil/dual
OUT_DIR=/data/luochunhua/od/mmdetection/work_dirs

#* train
CUDA_VISIBLE_DEVICES=$GPU_ID python /data/luochunhua/od/mmdetection/tools/train.py $CONFIG_DIR/$NAME.py

#* test
CUDA_VISIBLE_DEVICES=$GPU_ID python /data/luochunhua/od/mmdetection/tools/test.py $OUT_DIR/$NAME/$NAME.py $OUT_DIR/$NAME/latest.pth --eval bbox

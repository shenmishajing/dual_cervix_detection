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


CUDA_VISIBLE_DEVICES=2 python tools/test.py work_dirs/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil.py \
    work_dirs/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/epoch_9.pth \
    --out work_dirs/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/result.pkl

python tools/analysis_tools/analyze_results.py \
    work_dirs/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil.py \
    work_dirs/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/result.pkl \
    work_dirs/dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/show

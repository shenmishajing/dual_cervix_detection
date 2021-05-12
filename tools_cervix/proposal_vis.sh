#!/bin/bash

ACID=/data2/luochunhua/od/mmdetection/data/cervix/hsil_annos/test_acid.json
IODINE=/data2/luochunhua/od/mmdetection/data/cervix/hsil_annos/test_iodine.json

ARRAY=(dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_gamma3_acid_hsil)
    # dual_faster_rcnn_r50_fpn_droi_2x_acid_hsil \
    #    dual_faster_rcnn_r50_fpn_droi_2x_iodine_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnalignfusecat_2x_acid_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnalignfusecat_2x_iodine_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnalignfusecat_noatt_2x_acid_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnalignfusecat_noatt_2x_iodine_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnfusecat_2x_acid_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnfusecat_2x_iodine_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_acid_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_iodine_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_noatt_2x_acid_hsil \
    #     dual_faster_rcnn_r50_fpn_droi_noatt_2x_iodine_hsil)

WORK_DIRS=/data2/luochunhua/od/mmdetection/work_dirs

for EXP_NAME in ${ARRAY[@]}        
do
    echo $EXP_NAME
    #- replace code
    python tools_cervix/replace_code.py --exp_name $EXP_NAME

    #- generate proposals
    EXP_DIR=$WORK_DIRS/$EXP_NAME
    CONFIG=$EXP_DIR/$EXP_NAME".py"
    WEIGHT=$EXP_DIR/epoch_22.pth

    CUDA_VISIBLE_DEVICES=0 python tools/test.py \
        $CONFIG \
        $WEIGHT \
        --eval bbox


    #- visualize
    PROPOSAL_VIS_DIR=$EXP_DIR/proposal_vis
    if [ -d $PROPOSAL_VIS_DIR ];then
        rm -rf $PROPOSAL_VIS_DIR
    fi 

    mkdir $PROPOSAL_VIS_DIR

    if [[ $EXP_NAME =~ "acid" ]]; 
    then 
        PRIM="acid"
    else
        PRIM="iodine"
    fi

    python tools_cervix/proposal_vis.py \
        --p_dir $EXP_DIR/proposals \
        --prim $PRIM \
        --acid_data_json $ACID \
        --iodine_data_json $IODINE \
        --out_dir $PROPOSAL_VIS_DIR \
        --n_to_show 100 \
        --p_per_img 4 \
        --n_vis_img 10

done
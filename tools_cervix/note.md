- 总的可能的，输入是碘和醋酸同时，输出结果三种情况，只有碘，只有醋酸，醋酸和碘都有
- DualCervixDataset中的evaluation部分需要支持只有醋酸的检测结果，只有碘的检测结果，和两者都有的，三种情况进行评估
- Pipeline部分中的DualCervixDefaultFormatBundle可以修改传递到后面数据字典结构
- 各个函数涉及到输入参数设计


```
two_stage_detector
    
    foward_train:
        extract_feat(backbone + neck): acid, iodine
        rpn.forward_train 可能对碘、醋酸分别使用，也可能只对醋酸使用
        roi -> loss 可能只有醋酸的损失，也有可能碘和醋酸都有
    

```


python tools/analysis_tools/analyze_logs.py plot_curve /data/luochunhua/od/mmdetection/work_dirs/cascade_rcnn_r50_fpn_2x_acid_hsil/20210415_042002.log.json   /data/luochunhua/od/mmdetection/work_dirs/cascade_rcnn_r50_fpn_2x_iodine_hsil/20210415_101336.log.json /data/luochunhua/od/mmdetection/work_dirs/faster_rcnn_r50_fpn_2x_acid_hsil/20210415_040700.log.json /data/luochunhua/od/mmdetection/work_dirs/faster_rcnn_r50_fpn_2x_iodine_hsil/20210415_092358.log.json /data/luochunhua/od/mmdetection/work_dirs/reppoints_moment_r50_fpn_2x_acid_hsil/20210415_061359.log.json /data/luochunhua/od/mmdetection/work_dirs/reppoints_moment_r50_fpn_2x_iodine_hsil/20210415_103942.log.json /data/luochunhua/od/mmdetection/work_dirs/retinanet_r50_fpn_2x_acid_hsil/20210414_213817.log.json /data/luochunhua/od/mmdetection/work_dirs/retinanet_r50_fpn_2x_iodine_hsil/20210415_061942.log.json --keys bbox_mAP --legend cascade_rcnn_r50_fpn_2x_acid_hsil cascade_rcnn_r50_fpn_2x_iodine_hsil faster_rcnn_r50_fpn_2x_acid_hsil faster_rcnn_r50_fpn_2x_iodine_hsil reppoints_moment_r50_fpn_2x_acid_hsil reppoints_moment_r50_fpn_2x_iodine_hsil retinanet_r50_fpn_2x_acid_hsil retinanet_r50_fpn_2x_iodine_hsil --out single_map.png


python tools/analysis_tools/analyze_logs.py plot_curve /data/luochunhua/od/mmdetection/work_dirs/cascade_rcnn_r50_fpn_2x_acid_hsil/20210415_042002.log.json    --keys bbox_mAP --legend cascade_rcnn_r50_fpn_2x_acid_hsil 

python tools_cervix/analyze_logs.py plot_curve work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_2x_acid_hsil/20210422_083406.log.json work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_2x_iodine_hsil/20210422_105349.log.json work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_noatt_2x_acid_hsil/20210424_131843.log.json work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_noatt_2x_iodine_hsil/20210424_132010.log.json  --keys bbox_mAP --legend  fpn_droi_2x_acid_hsil fpn_droi_2x_iodine_hsil fpn_droi_noatt_2x_acid_hsil fpn_droi_noatt_2x_iodine_hsil --out ./stat/dual_dualdet_map1.png



python tools_cervix/analyze_logs.py plot_curve work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_noatt_nooffset_2x_acid_hsil/20210426_112008.log.json work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_noatt_nooffset_2x_iodine_hsil/20210426_193112.log.json work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_nooffset_2x_acid_hsil/20210424_052254.log.json work_dirs/dual_dualdet_faster_rcnn_r50_fpn_droi_nooffset_2x_iodone_hsil/20210424_055618.log.json --keys bbox_mAP --legend   fpn_droi_noatt_nooffset_2x_acid_hsil fpn_droi_noatt_nooffset_2x_iodine_hsil fpn_droi_nooffset_2x_acid_hsil fpn_droi_nooffset_2x_iodone_hsil --out ./stat/dual_dualdet_map2.png
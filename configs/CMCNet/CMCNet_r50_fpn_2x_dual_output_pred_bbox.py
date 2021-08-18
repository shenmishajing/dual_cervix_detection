_base_ = [
    '../_base_/models/CMCNet_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

load_from = '/data/zhengwenhao/Result/DualCervixDetection/CMCNet/pred_bbox/faster_rcnn_late_fusion_r50_fpn_2x_dual_B1Id_17e/epoch_17.pth'

# model settings
model = dict(
    output_path = '/data/zhengwenhao/Result/DualCervixDetection/CMCNet/pred_bbox/faster_rcnn_late_fusion_r50_fpn_2x_dual_B1Id_17e/results',
    roi_head = dict(bbox_head = dict(num_classes = 1)))

log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'dual_cervix_detection',
                                name = 'CMCNet_r50_fpn_2x_dual',
                                tags = ['mmdetection', 'CMCNet', 'r50', 'fpn', '2x', 'dual']))])

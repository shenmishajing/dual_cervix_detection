_base_ = [
    '../_base_/models/faster_rcnn_dual_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    prim = 'acid',
    roi_head = dict(
        enlarge = 0,
        bbox_head = dict(num_classes = 1)))

load_from = './work_dirs/faster_rcnn_dual_r50_fpn_1x_dual_zEPyVcfph/latest.pth'

log_config = dict(
    interval = 50,
    hooks = [
        dict(type = 'TextLoggerHook'),
        dict(type = 'WandbLoggerHook',
             with_step = False,
             init_kwargs = dict(project = 'dual_cervix_detection',
                                name = 'faster_rcnn_dual_r50_fpn_2x_acid',
                                tags = ['mmdetection', 'faster_rcnn_dual', 'r50', 'fpn', '2x', 'acid']))])

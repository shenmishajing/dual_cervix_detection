_base_ = [
    '../_base_/models/faster_rcnn_early_fusion_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/dual/dual_base.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(roi_head = dict(bbox_head = dict(num_classes = 1)))

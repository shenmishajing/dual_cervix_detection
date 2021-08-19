_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/dual_cervix/hsil/acid/acid_base.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    prim = 'acid',
    roi_head = dict(bbox_head = dict(num_classes = 1)))

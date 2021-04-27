_base_ = [
    './dual_dualdet_faster_rcnn_r50_fpn_droi_fpnalignfusecat_2x_iodine_hsil.py'
]

model = dict(
    roi_head=dict(
        attention_cfg=None
     ))
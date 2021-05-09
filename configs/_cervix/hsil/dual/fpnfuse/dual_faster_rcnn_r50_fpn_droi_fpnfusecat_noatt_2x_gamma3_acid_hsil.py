_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_acid_hsil.py'
]

model = dict(
    roi_head=dict(
        bridge_bbox_roi_extractor=dict(
            #! spatial_scale 会根据featmap_strides自动添加的，gamma不知道啥意思
            roi_layer=dict(gamma=0.3),
     ))
)
_base_ = [
    './dual_faster_rcnn_r50_fpn_droi_dualfpnfusecat_noatt_2x_acid_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            in_channels=256 * 3, 
            out_channels=256 * 3, 
            with_conv=False, 
        ),
        offset_cfg=dict(
            type="ProposalOffsetXY",
            in_channels=256 * 3, # 根据with_conv改变
            out_channels=256, 
            roi_feat_area=7*7), #! squared roi_feat_size(in bbox_head)
     ))
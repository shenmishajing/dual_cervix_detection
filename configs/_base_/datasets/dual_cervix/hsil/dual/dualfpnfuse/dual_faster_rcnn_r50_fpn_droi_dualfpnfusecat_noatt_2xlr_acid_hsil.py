_base_ = [
    '../fpnfuse/dual_faster_rcnn_r50_fpn_droi_fpnfusecat_noatt_2x_acid_hsil.py'
]

model = dict(
    roi_head=dict(
        fpn_fuser_cfg=dict(
            type="DualFPNFeatureFuser",
            roi_feat_size=7,
            num_levels=4,
            in_channels=256 * 3, 
            out_channels=256, 
            with_conv=True, 
            fuse_type="cat", 
            naive_fuse=True #! must be true
        ),
        attention_cfg=None,
        offset_cfg=dict(
            type="ProposalOffsetXY",
            in_channels=256, 
            out_channels=256, 
            roi_feat_area=7*7), #! squared roi_feat_size(in bbox_head)
     ))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 16])
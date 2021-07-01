# 加入新的模块需要重新编译一下， 新的模块才会有效
- python setup.py develop

- faster-rcnn
CUDA_VISIBLE_DEVICES=5 python tools/test.py \
    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    --out /data/luochunhua/od/mmdetection/test_output/output.pkl \
    --eval bbox  \
    --show-dir test_output/anno_image \
    --show-score-thr 0.5

- reppoints
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py \
    checkpoints/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco_20200329-4b38409a.pth \
    --out /data/luochunhua/od/mmdetection/test_output/output.pkl \
    --eval bbox  \
    --show-dir test_output/anno_image \
    --show-score-thr 0.5

CUDA_VISIBLE_DEVICES=8 python tools/train.py \
    configs/reppoints/reppoints_moment_r50_fpn_gn-neck+head_1x_coco.py     



```python
# img_meta: 
{   
    'filename': 'data/coco/train2017/000000251577.jpg',
    'ori_filename': '000000251577.jpg',
    'ori_shape': (289, 500, 3),
    'img_shape': (770, 1333, 3),
    'pad_shape': (800, 1344, 3),
    'scale_factor': array([2.666    , 2.6643598, 2.666    , 2.6643598], dtype=float32),
    'flip': False,
    'flip_direction': None, 
    'img_norm_cfg': 
        {
            'mean': array([123.675, 116.28 , 103.53 ], dtype=float32),
            'std': array([58.395, 57.12 , 57.375], dtype=float32),
            'to_rgb': True
        },
    'batch_input_shape': (800, 1344)
}
```
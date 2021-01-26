from mmdet.apis import init_detector, inference_detector
import mmcv
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_file = "/data/luochunhua/od/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "/data/luochunhua/od/mmdetection/checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = "/data/luochunhua/od/mmdetection/demo/demo.jpg"
result = inference_detector(model, img)

# model.show_result(img, result)
model.show_result(img, result, out_file='result.jpg')

# video = mmcv.VideoReader("video.mp4")
# for frame in video:
    # result = inference_detector(model, frame)
    # model.show_result(frame, result, wait_time=1)

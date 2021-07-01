#!/bin/sh
# 将coco数据软链接到data文件夹下

cd /data/luochunhua/od/mmdetection240/
mkdir ./data
mkdir ./data/coco
ln -s /data/prince/dataset/COCO/2017/annotations_trainval2017/annotations ./data/coco/annotations
ln -s /data/prince/dataset/COCO/2017/Images/train2017 ./data/coco/train2017
ln -s /data/prince/dataset/COCO/2017/Images/val2017 ./data/coco/val2017
ln -s /data/prince/dataset/COCO/2017/Images/test2017 ./data/coco/test2017

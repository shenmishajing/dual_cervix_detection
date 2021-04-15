- 总的可能的，输入是碘和醋酸同时，输出结果三种情况，只有碘，只有醋酸，醋酸和碘都有
- DualCervixDataset中的evaluation部分需要支持只有醋酸的检测结果，只有碘的检测结果，和两者都有的，三种情况进行评估
- Pipeline部分中的DualCervixDefaultFormatBundle可以修改传递到后面数据字典结构
- 各个函数涉及到输入参数设计


```
two_stage_detector
    
    foward_train:
        extract_feat(backbone + neck): acid, iodine
        rpn.forward_train 可能对碘、醋酸分别使用，也可能只对醋酸使用
        roi -> loss 可能只有醋酸的损失，也有可能碘和醋酸都有
    

```
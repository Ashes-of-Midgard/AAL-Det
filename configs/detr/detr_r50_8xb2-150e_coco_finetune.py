_base_ = [
    './detr_r50_8xb2-150e_coco.py'
]

# dataset settings
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))

# learning policy
max_epochs = 1
train_cfg = dict(
    type='CustomizedTrainLoop', max_epochs=max_epochs, val_interval=1)

# finetune
load_from='https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

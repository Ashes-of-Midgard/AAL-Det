_base_ = [
    '../detr/detr_r50_8xb2-150e_coco.py'
]

# model settings
model = dict(
    type='DETRAAL',
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'),
    backbone = dict(type='ResNetCBAM'),
    neck=dict(type='ChannelMapperCBAM'))

# dataset settings
train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')
    ))

train_cfg = dict(type='AALTrainLoop', max_epochs=150, val_interval=5)
test_cfg = dict(type='AdvTestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/mAP'))
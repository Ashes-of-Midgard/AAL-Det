_base_ = [
    '../ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py'
]

# model settings
model = dict(
    type='SingleStageDetectorAAL',
    backbone=dict(
        _delete_=True,
        type='ResNetCBAM',
        depth=34,
        out_indices=(2,3)
    ),
    neck=dict(
        type='SSDNeckCBAM',
        in_channels=(256, 512),
        out_channels=(256, 512, 512, 256, 256, 128),
    ))

# training schedule
max_epochs = 150
train_cfg = dict(type='AALTrainLoop', max_epochs=max_epochs, val_interval=5)
test_cfg = dict(type='AdvTestLoop', vis_dir='visual')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/mAP'))

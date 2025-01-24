_base_ = ['../deformable_detr/deformable-detr_r50_16xb2-50e_coco.py']

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[111.89, 111.89, 111.89],
    std=[27.62, 27.62, 27.62],
    bgr_to_rgb=True,
    pad_size_divisor=1)
model = dict(
    data_preprocessor=data_preprocessor,
    num_queries=300,
    backbone=dict(depth=50),
    neck=dict(in_channels=[512, 1024, 2048]),
    bbox_head=dict(num_classes=1)
)

# dataset settings
input_size = (1000, 600)
data_root = 'data/open-sirst-v2'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=data_preprocessor['mean'],
        to_rgb=data_preprocessor['bgr_to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=input_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=input_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=8,
    batch_sampler=None,
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='SirstDataset',
            split='train_full',
            data_root=data_root,
            pipeline=train_pipeline)))
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    dataset=dict(
        type='SirstDataset',
        split='val_full',
        data_root=data_root,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# learning policy
max_epochs = 40
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='AdvTestLoop')

val_evaluator = dict(_delete_=True, type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='pascal_voc/mAP'))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[50],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)

# finetune
load_from='https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr_r50_16xb2-50e_coco/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth'
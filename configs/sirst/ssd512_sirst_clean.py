_base_ = [
    '../ssd/ssd512_coco.py'
]

# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[111.89, 111.89, 111.89],
    std=[27.62, 27.62, 27.62],
    bgr_to_rgb=True,
    pad_size_divisor=1)
model = dict(
    type='SingleStageDetector',
    data_preprocessor=data_preprocessor,
    bbox_head=dict(
        num_classes=1
        )
    )

# dataset settings
input_size = (512, 512)
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
    batch_size=24,
    num_workers=4,
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
    batch_size=8,
    dataset=dict(
        type='SirstDataset',
        split='val_full',
        data_root=data_root,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# train_cfg = dict(type='AdvTrainLoop', val_interval=1)
train_cfg = dict(max_epochs=150, val_interval=5)
test_cfg = dict(type='AdvTestLoop')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(type='SGD', lr=3e-4, momentum=0.9, weight_decay=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=150,
        by_epoch=True,
        milestones=[80, 120],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='pascal_voc/mAP'))

val_evaluator = dict(_delete_=True, type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

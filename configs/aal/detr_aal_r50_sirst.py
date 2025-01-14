_base_ = [
    '../detr/detr_r50_8xb2-150e_coco.py'
]

# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[111.89, 111.89, 111.89],
    std=[27.62, 27.62, 27.62],
    bgr_to_rgb=True,
    pad_size_divisor=1)
model = dict(
    type='DETRAAL',
    init_cfg=dict(type='Pretrained', checkpoint='../models/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'),
    data_preprocessor=data_preprocessor,
    neck=dict(type='ChannelMapperCBAM'),
    bbox_head=dict(num_classes=1))

# dataset settings
input_size = (1000, 600)
data_root = 'data/open-sirst-v2'
train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    batch_sampler=None,
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='SirstDataset',
            split='train_full',
            data_root=data_root,
            pipeline={{_base_.train_pipeline}})))
val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    dataset=dict(
        type='SirstDataset',
        split='val_full',
        data_root=data_root,
        pipeline={{_base_.test_pipeline}}))
test_dataloader = val_dataloader

train_cfg = dict(type='AALTrainLoop', max_epochs=150, val_interval=5)
test_cfg = dict(type='AdvTestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='pascal_voc/mAP'))

val_evaluator = dict(_delete_=True, type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator
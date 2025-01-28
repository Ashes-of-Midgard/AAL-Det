_base_ = ['../yolo/yolov3_d53_8xb8-ms-608-273e_coco.py']

# model settings
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[0, 0, 0],
    std=[255., 255., 255.],
    bgr_to_rgb=True,
    pad_size_divisor=32)
model = dict(
    type='YOLODetector',
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(_delete_=True)),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=20))

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
input_size = (608, 608)
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
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
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
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=3,
    dataset=dict(  # RepeatDataset
        # the dataset is repeated 10 times, and the training schedule is 2x,
        # so the actual epoch = 12 * 10 = 120.
        times=1,
        dataset=dict(  # ConcatDataset
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/train.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline)
                # dict(
                #     type=dataset_type,
                #     data_root=data_root,
                #     ann_file='VOC2012/ImageSets/Main/train.txt',
                #     data_prefix=dict(sub_data_root='VOC2012/'),
                #     filter_cfg=dict(filter_empty_gt=True, min_size=32),
                #     pipeline=train_pipeline)
            ])))
val_dataloader = dict(dataset=dict(ann_file='VOC2007/ImageSets/Main/val.txt',
                                   pipeline=test_pipeline))
test_dataloader = val_dataloader

# training schedule
max_epochs = 24
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
test_cfg = dict(type='AdvTestLoop')
# test_cfg = dict(type='AdvTestLoop', vis_dir='visual')

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[10, 15], gamma=0.1)
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='pascal_voc/mAP'))

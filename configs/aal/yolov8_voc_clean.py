_base_ = ['../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py', '../_base_/datasets/voc0712.py']

# model settings
deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768

affine_scale = 0.9
mixup_prob = 0.1

norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
num_classes = 20
strides = [8, 16, 32]

loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4


tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=False,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

data_preprocessor=dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[1, 1, 1],
    bgr_to_rgb=True,
    pad_size_divisor=1)

model = dict(
    type='YOLODetector',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoderYOLO'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULossYOLO',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg)

# dataset setting
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
input_size = (640, 640)
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

# optimizer settings
max_epochs = 24
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
test_cfg = dict(type='AdvTestLoop')
# test_cfg = dict(type='AdvTestLoop', vis_dir='visual')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[10, 15],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='pascal_voc/mAP'))
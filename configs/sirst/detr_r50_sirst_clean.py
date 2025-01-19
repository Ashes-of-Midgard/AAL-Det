_base_ = [
    '../detr/detr_r50_8xb2-150e_coco.py'
]

# model settings
# data_preprocessor = dict(
#     type='DetDataPreprocessor',
#     mean=[111.89, 111.89, 111.89],
#     std=[27.62, 27.62, 27.62],
#     bgr_to_rgb=True,
#     pad_size_divisor=1)
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmdetection/v3.0/detr/detr_r50_8xb2-150e_coco/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'),
    # data_preprocessor=data_preprocessor,
    bbox_head=dict(num_classes=80))

# dataset settings
# input_size = (1000, 600)
# data_root = 'data/open-sirst-v2'

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Expand',
#         mean=data_preprocessor['mean'],
#         to_rgb=data_preprocessor['bgr_to_rgb'],
#         ratio_range=(1, 4)),
#     dict(
#         type='MinIoURandomCrop',
#         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
#         min_crop_size=0.3),
#     dict(type='Resize', scale=input_size, keep_ratio=False),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=input_size, keep_ratio=False),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]

# train_dataloader = dict(
#     _delete_=True,
#     batch_size=2,
#     num_workers=2,
#     batch_sampler=None,
#     dataset=dict(
#         type='RepeatDataset',
#         times=3,
#         dataset=dict(
#             type='SirstDataset',
#             split='train_full',
#             data_root=data_root,
#             pipeline=train_pipeline)))
# val_dataloader = dict(
#     _delete_=True,
#     batch_size=1,
#     dataset=dict(
#         type='SirstDataset',
#         split='val_full',
#         data_root=data_root,
#         pipeline=test_pipeline))
# test_dataloader = val_dataloader

train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/')))

# train_cfg = dict(type='AdvTrainLoop', max_epochs=150, val_interval=1)
# test_cfg = dict(type='AdvTestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/mAP'))

# val_evaluator = dict(_delete_=True, type='VOCMetric', metric='mAP', eval_mode='11points')
# test_evaluator = val_evaluator
_base_ = ['./deformable_detr_r50_voc_clean.py']

model = dict(
    type='DeformableDETRAAL',
    backbone=dict(type='ResNetCBAM'),
    neck=dict(type='ChannelMapperCBAM')
)

train_cfg = dict(type='AALTrainLoop')
_base_ = ['../sirst/deformable_detr_r50_sirst_clean.py']

model = dict(
    type='DeformableDETRAAL',
    backbone=dict(type='ResNetCBAM'),
    neck=dict(type='ChannelMapperCBAM')
)

train_cfg = dict(type='AALTrainLoop')
_base_ = [
    '../sirst/detr_r50_sirst_clean.py'
]

model = dict(
    type='DETRAAL',
    backbone=dict(type='ResNetCBAM'),
    neck=dict(type='ChannelMapperCBAM')
)

train_cfg = dict(type='AALTrainLoop')

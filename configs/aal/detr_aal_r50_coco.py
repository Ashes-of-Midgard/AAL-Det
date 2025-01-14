_base_ = [
    '../detr/detr_r50_8xb2-150e_coco.py'
]

# model settings
model = dict(
    type='DETRAAL',
    init_cfg=dict(type='Pretrained'),
    neck=dict(type='ChannelMapperCBAM'))

# dataset settings
train_cfg = dict(type='AALTrainLoop', max_epochs=150, val_interval=5)
test_cfg = dict(type='AdvTestLoop')
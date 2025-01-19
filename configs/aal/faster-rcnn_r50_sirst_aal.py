_base_ = [
    '../sirst/faster-rcnn_r50_sirst_clean.py'
]

model = dict(
    type='TwoStageDetectorAAL',
    backbone=dict(type='ResNetCBAM'),
    neck=dict(type='FPNCBAM'))

train_cfg = dict(type='AALTrainLoop')

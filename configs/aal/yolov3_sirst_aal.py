_base_ = ['../sirst/yolov3_sirst_clean.py']

# model settings
model = dict(
    type='YOLODetectorAAL',
    backbone=dict(type='DarknetCBAM'),
    neck=dict(type='YOLOV3NeckCBAM')
)

# training schedule
train_cfg = dict(type='AALTrainLoop')

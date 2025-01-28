_base_ = ['./yolov3_voc_clean.py']

model = dict(
    type='YOLODetectorAAL',
    # backbone=dict(type='DarknetCBAM'),
    neck=dict(type='YOLOV3NeckCBAM')
)

train_cfg = dict(type='AALTrainLoop')
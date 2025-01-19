_base_ = ['../sirst/yolov8_sirst_clean.py']

model = dict(
    type='YOLODetectorAAL',
    backbone=dict(
        type='YOLOv8CSPDarknetCBAM'
    ),
    neck=dict(
        type='YOLOv8PAFPNCBAM'
    ),
)

# optimizer settings
train_cfg = dict(type='AALTrainLoop')
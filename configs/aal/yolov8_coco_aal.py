_base_ = ['./yolov8_coco_clean.py']

model = dict(
    type='YOLODetectorAAL',
    backbone=dict(
        type='YOLOv8CSPDarknetCBAM'
    ),
    neck=dict(
        type='YOLOv8PAFPNCBAM'
    ),
)
train_cfg = dict(type='AALTrainLoop')
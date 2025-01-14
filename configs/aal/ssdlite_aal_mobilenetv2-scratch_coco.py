_base_ = [
    '../ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py'
]

# model settings
model = dict(
    type='SingleStageDetectorAAL',
    neck=dict(
        type='SSDNeckCBAM',
    ))

# training schedule
max_epochs = 150
train_cfg = dict(type='AALTrainLoop', max_epochs=max_epochs, val_interval=5)
test_cfg = dict(type='AdvTestLoop', vis_dir='visual')

_base_ = [
    './faster-rcnn_r50_sirst_clean.py'
]

train_cfg = dict(type='AdvTrainLoop')

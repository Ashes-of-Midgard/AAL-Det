_base_ = ['../yolo/yolov3_d53_8xb8-ms-608-273e_coco.py']

# training schedule
max_epochs = 150
train_cfg = dict(type='AdvTrainLoop', max_epochs=max_epochs, val_interval=5)
test_cfg = dict(type='AdvTestLoop', vis_dir='visual')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/mAP'))

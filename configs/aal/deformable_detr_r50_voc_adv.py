_base_ = ['./deformable_detr_r50_voc_clean.py']

train_cfg = dict(type='AdvTrainLoop')
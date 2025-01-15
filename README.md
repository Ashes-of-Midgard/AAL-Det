# Install
```
conda create -n aal-det python=3.8
conda activate aal-det
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
cd aal-det
pip install -e . -v
```

# TRAIN

```shell
python tools/train.py configs/aal/yolov3_aal_d53_sirst.py
python tools/test.py configs/aal/yolov3_aal_d53_sirst.py path/to/model.pth
```

# SIRST

||mAP(clean)|mAP(FGSM)|
|---|---|---|
|SSD-FGSM|0.224|0.233|
|SSD-AAL|0.192|0.225|
|Faster-RCNN-FGSM|||
|Faster-RCNN-AAL|||
|YOLOv3-FGSM|0.465|0.444|
|YOLOv3-AAL|0.503|0.505|
|Oscar-FGSM|||
|Oscar-AAL|||
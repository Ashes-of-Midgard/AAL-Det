# Install
```
conda create -n aal-det python=3.8
conda activate aal-det
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
cd aal-det
pip install -e . -v
```

# Dataset
For COCO dataset, you can just download official coco dataset and extract in ```data/coco```

For Single-frame InfraRed Small Target(SIRST) dataset, run
```shell
cd data
git clone https://github.com/YimianDai/open-sirst-v2.git
```

# TRAIN
COCO
```shell
python tools/train.py configs/aal/yolov8_coco_aal.py
python tools/test.py configs/aal/yolov8_coco_aal.py path/to/model.pth
```

SIRST
```shell
python tools/train.py configs/aal/yolov8_sirst_aal.py
python tools/test.py configs/aal/yolov8_sirst_aal.py path/to/model.pth
```

# SIRST

||mAP(clean)|mAP(FGSM)|
|---|---|---|
|YOLOv8-Clean|0.515|0.508|
|YOLOv8-FGSM|0.535|0.489|
|YOLOv8-AAL|0.556|0.524|

# COCO
||mAP(clean)|mAP(FGSM)|
|---|---|---|
|YOLOv8-Clean|||
|YOLOv8-FGSM|||
|YOLOv8-AAL|||
# Implementation

![alt text](image.png)

AAL method is supposed to improve trained models' test accuracy when the samples are perturbed by adversarial attack.

Available training and testing config files:

**YOLOv8**: ```AAL-Det/configs/aal/yolov8_coco_aal.py```, ```AAL-Det/configs/aal/yolov8_sirst_aal.py```

The AAL training loops are implemented in ```AAL-Det/mmdet/engine/runner/loops_adv.py```.

NOTE: Although the complete AAL training requires backtracking process, we find that it unneccesary for detection models by experiments.
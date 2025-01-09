# Install
```
conda create -n aal-det python=3.8
conda activate aal-det
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install mmdet==3.3.0
cd aal-det
pip install -e . -v
```


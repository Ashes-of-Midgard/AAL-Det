python tools/train.py configs/aal/deformable_detr_r50_voc_clean.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_voc_clean/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_voc_clean/2025*")
mv work_dirs/deformable_detr_r50_voc_clean/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_voc_clean/

python tools/train.py configs/aal/deformable_detr_r50_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_voc_aal/2025*")
mv work_dirs/deformable_detr_r50_voc_aal/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_voc_aal/

python tools/train.py configs/aal/yolov3_voc_clean.py
result=$(find /root/AAL-Det/work_dirs/yolov3_voc_clean/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_voc_clean/2025*")
mv work_dirs/yolov3_voc_clean/best_* $result
mv $result work_dirs/saved/yolov3_voc_clean/

python tools/train.py configs/aal/yolov3_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov3_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_voc_aal/2025*")
mv work_dirs/yolov3_voc_aal/best_* $result
mv $result work_dirs/saved/yolov3_voc_aal/

python tools/train.py configs/aal/yolov8_voc_clean.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_clean/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_clean/2025*")
mv work_dirs/yolov8_voc_clean/best_* $result
mv $result work_dirs/saved/yolov8_voc_clean/

python tools/train.py configs/aal/yolov8_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_aal/2025*")
mv work_dirs/yolov8_voc_aal/best_* $result
mv $result work_dirs/saved/yolov8_voc_aal/

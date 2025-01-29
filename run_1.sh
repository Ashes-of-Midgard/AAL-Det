python tools/train.py configs/aal/yolov8_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_aal/2025*")
mv work_dirs/yolov8_voc_aal/best_* $result
mv $result work_dirs/saved/yolov8_voc_aal/

python tools/train.py configs/aal/yolov8_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_aal/2025*")
mv work_dirs/yolov8_voc_aal/best_* $result
mv $result work_dirs/saved/yolov8_voc_aal/

python tools/train.py configs/aal/yolov8_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_aal/2025*")
mv work_dirs/yolov8_voc_aal/best_* $result
mv $result work_dirs/saved/yolov8_voc_aal/

python tools/train.py configs/aal/yolov8_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_aal/2025*")
mv work_dirs/yolov8_voc_aal/best_* $result
mv $result work_dirs/saved/yolov8_voc_aal/

python tools/train.py configs/aal/yolov8_voc_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov8_voc_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov8_voc_aal/2025*")
mv work_dirs/yolov8_voc_aal/best_* $result
mv $result work_dirs/saved/yolov8_voc_aal/
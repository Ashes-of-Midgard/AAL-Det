python tools/train.py configs/aal/yolov3_sirst_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov3_sirst_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_sirst_aal/2025*")
mv work_dirs/yolov3_sirst_aal/best_* $result
mv $result work_dirs/saved/yolov3_sirst_aal/

python tools/train.py configs/aal/yolov3_sirst_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov3_sirst_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_sirst_aal/2025*")
mv work_dirs/yolov3_sirst_aal/best_* $result
mv $result work_dirs/saved/yolov3_sirst_aal/

python tools/train.py configs/aal/yolov3_sirst_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov3_sirst_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_sirst_aal/2025*")
mv work_dirs/yolov3_sirst_aal/best_* $result
mv $result work_dirs/saved/yolov3_sirst_aal/

python tools/train.py configs/aal/yolov3_sirst_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov3_sirst_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_sirst_aal/2025*")
mv work_dirs/yolov3_sirst_aal/best_* $result
mv $result work_dirs/saved/yolov3_sirst_aal/

python tools/train.py configs/aal/yolov3_sirst_aal.py
result=$(find /root/AAL-Det/work_dirs/yolov3_sirst_aal/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/yolov3_sirst_aal/2025*")
mv work_dirs/yolov3_sirst_aal/best_* $result
mv $result work_dirs/saved/yolov3_sirst_aal/
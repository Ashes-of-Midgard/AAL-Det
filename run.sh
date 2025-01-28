python tools/train.py configs/sirst/deformable_detr_r50_sirst_adv.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/2025*")
mv work_dirs/deformable_detr_r50_sirst_adv/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_sirst_adv/

python tools/train.py configs/sirst/deformable_detr_r50_sirst_adv.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/2025*")
mv work_dirs/deformable_detr_r50_sirst_adv/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_sirst_adv/

python tools/train.py configs/sirst/deformable_detr_r50_sirst_adv.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/2025*")
mv work_dirs/deformable_detr_r50_sirst_adv/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_sirst_adv/

python tools/train.py configs/sirst/deformable_detr_r50_sirst_adv.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/2025*")
mv work_dirs/deformable_detr_r50_sirst_adv/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_sirst_adv/

python tools/train.py configs/sirst/deformable_detr_r50_sirst_adv.py
result=$(find /root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/ -maxdepth 1 -type d -path "/root/AAL-Det/work_dirs/deformable_detr_r50_sirst_adv/2025*")
mv work_dirs/deformable_detr_r50_sirst_adv/best_* $result
mv $result work_dirs/saved/deformable_detr_r50_sirst_adv/
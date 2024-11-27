import os
import torch
print(torch.cuda.is_available())  

YOLO_PATH = "yolov5"


train_cmd = f"""
python {YOLO_PATH}/train.py --img 1280 --batch 16 --epochs 100 \
--data {YOLO_PATH}/data.yaml --weights yolov5n.pt --project yolov5_runs
"""
os.system(train_cmd)
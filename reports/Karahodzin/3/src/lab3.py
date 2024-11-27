import os
import pandas as pd
from pathlib import Path
import shutil  
from PIL import Image  


DATASET_PATH = "rtsd-d3-gt"       
IMAGES_PATH = "rtsd-d3-frames"    
OUTPUT_PATH = "yolo_data"         


CLASS_MAPPING = {
    "blue_border": 0,
    "blue_rect": 1,
    "main_road": 2
}


os.makedirs(OUTPUT_PATH, exist_ok=True)

def convert_to_yolo(row, img_width, img_height, class_id):
    """
    Конвертирует аннотацию в формат YOLO.
    """
    x_center = (row['x_from'] + row['width'] / 2) / img_width
    y_center = (row['y_from'] + row['height'] / 2) / img_height
    width = row['width'] / img_width
    height = row['height'] / img_height

    
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        return None  

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


for group, class_id in CLASS_MAPPING.items():
    print(f"Processing group: {group}")
    for split in ["train", "test"]:
        csv_file = os.path.join(DATASET_PATH, group, f"{split}_gt.csv")
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            continue

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_path = os.path.join(IMAGES_PATH, split, row['filename'])
            if not os.path.exists(img_path):
                continue

            
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                continue
            
            
            yolo_annotation = convert_to_yolo(row, img_width, img_height, class_id)
            if yolo_annotation is None:
                print(f"Skipping invalid annotation for image {row['filename']}")
                continue

            
            output_file = os.path.join(OUTPUT_PATH, split, "labels", row['filename'].replace('.jpg', '.txt'))
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'a') as f:
                f.write(yolo_annotation + '\n')

            
            output_image_dir = os.path.join(OUTPUT_PATH, split, "images")
            os.makedirs(output_image_dir, exist_ok=True)
            shutil.copy(img_path, output_image_dir)

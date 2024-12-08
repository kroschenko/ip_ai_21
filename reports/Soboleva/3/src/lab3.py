

# import os
# import pandas as pd
# from pathlib import Path
# import shutil  
# from PIL import Image  


# DATASET_PATH = "rtsd-d3-gt"       
# IMAGES_PATH = "rtsd-d3-frames"    
# OUTPUT_PATH = "yolo_data"         


# CLASS_MAPPING = {
#     "blue_border": 0,
#     "blue_rect": 1,
#     "main_road": 2
# }


# os.makedirs(OUTPUT_PATH, exist_ok=True)

# def convert_to_yolo(row, img_width, img_height, class_id):
#     """
#     Конвертирует аннотацию в формат YOLO.
#     """
#     x_center = (row['x_from'] + row['width'] / 2) / img_width
#     y_center = (row['y_from'] + row['height'] / 2) / img_height
#     width = row['width'] / img_width
#     height = row['height'] / img_height

    
#     if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
#         return None  

#     return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# for group, class_id in CLASS_MAPPING.items():
#     print(f"Processing group: {group}")
#     for split in ["train", "test"]:
#         csv_file = os.path.join(DATASET_PATH, group, f"{split}_gt.csv")
#         if not os.path.exists(csv_file):
#             print(f"File not found: {csv_file}")
#             continue

#         df = pd.read_csv(csv_file)

#         for _, row in df.iterrows():
#             img_path = os.path.join(IMAGES_PATH, split, row['filename'])
#             if not os.path.exists(img_path):
#                 continue

            
#             try:
#                 with Image.open(img_path) as img:
#                     img_width, img_height = img.size
#             except Exception as e:
#                 print(f"Error reading image {img_path}: {e}")
#                 continue
            
            
#             yolo_annotation = convert_to_yolo(row, img_width, img_height, class_id)
#             if yolo_annotation is None:
#                 print(f"Skipping invalid annotation for image {row['filename']}")
#                 continue

            
#             output_file = os.path.join(OUTPUT_PATH, split, "labels", row['filename'].replace('.jpg', '.txt'))
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
#             with open(output_file, 'a') as f:
#                 f.write(yolo_annotation + '\n')

            
#             output_image_dir = os.path.join(OUTPUT_PATH, split, "images")
#             os.makedirs(output_image_dir, exist_ok=True)
#             shutil.copy(img_path, output_image_dir)


import os
import pandas as pd
from pathlib import Path
import shutil
from PIL import Image

# Пути к данным
DATASET_PATH = "rtsd-d3-gt/blue_rect"  # Путь к аннотациям группы blue_rect
IMAGES_PATH = "rtsd-d3-frames"         # Путь к изображениям
OUTPUT_PATH = "yolo_data"              # Путь для сохранения данных в формате YOLO

CLASS_ID = 1  # Класс blue_rect

# Создать выходную папку, если её нет
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Отображение классов из data.yaml на их индексы
CLASS_MAPPING = {
    '5_20': 0, '5_19_1': 1, '5_15_5': 2, '6_3_1': 3, '6_7': 4, 
    '5_15_3': 5, '6_4': 6, '6_6': 7, '5_15_1': 8, '5_15_2': 9,
    '5_6': 10, '5_5': 11, '5_15_2_2': 12, '5_22': 13, '5_3': 14,
    '6_2_n50': 15, '6_2_n70': 16, '5_15_7': 17, '5_14': 18, 
    '5_21': 19, '6_2_n60': 20, '5_7_1': 21, '5_7_2': 22, 
    '5_11': 23, '5_8': 24
}

def convert_to_yolo(row, img_width, img_height):
    """
    Конвертирует аннотацию в формат YOLO.
    """
    x_center = (row['x_from'] + row['width'] / 2) / img_width
    y_center = (row['y_from'] + row['height'] / 2) / img_height
    width = row['width'] / img_width
    height = row['height'] / img_height

    # Проверка на корректность координат
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        return None

    # Получить индекс класса
    class_id = CLASS_MAPPING.get(row['sign_class'])
    if class_id is None:
        print(f"Unknown sign_class: {row['sign_class']}")
        return None
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Обработка train и test данных
for split in ["train", "test"]:
    csv_file = os.path.join(DATASET_PATH, f"{split}_gt.csv")
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        continue

    # Чтение аннотаций
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_PATH, split, row['filename'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Открытие изображения для получения размеров
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        # Конвертация аннотации в YOLO-формат
        yolo_annotation = convert_to_yolo(row, img_width, img_height)
        print(yolo_annotation)
        if yolo_annotation is None:
            print(f"Skipping invalid annotation for image {row['filename']}")
            continue

        # Сохранение аннотации
        output_label_file = os.path.join(OUTPUT_PATH, split, "labels", row['filename'].replace('.jpg', '.txt'))
        os.makedirs(os.path.dirname(output_label_file), exist_ok=True)
        with open(output_label_file, 'a') as f:
            f.write(yolo_annotation + '\n')

        # Копирование изображения
        output_image_dir = os.path.join(OUTPUT_PATH, split, "images")
        os.makedirs(output_image_dir, exist_ok=True)
        shutil.copy(img_path, output_image_dir)

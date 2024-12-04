import os
import pandas as pd
from PIL import Image

def convert_csv_to_yolo(csv_path, images_dir, labels_dir):
    # Читаем CSV
    data = pd.read_csv(csv_path)

    for _, row in data.iterrows():
        filename = row['filename']
        x_from, y_from, width, height = row['x_from'], row['y_from'], row['width'], row['height']
        class_name = row['sign_class']  # Пример: "3_1", "3_24_n20"

        # Конвертируем класс в числовой ID
        class_id = convert_class_to_id(class_name)

        # Получаем размеры изображения
        image_path = os.path.join(images_dir, filename)
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Конвертируем координаты в YOLO формат
        x_center = (x_from + width / 2) / img_width
        y_center = (y_from + height / 2) / img_height
        w_norm = width / img_width
        h_norm = height / img_height

        # Формируем строку для файла
        yolo_line = f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n"

        # Путь к файлу аннотаций
        label_path = os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")
        
        # Проверяем, существует ли папка, если нет - создаём
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        # Открываем файл на добавление, если файл существует, и добавляем аннотации
        with open(label_path, 'a') as label_file:
            label_file.write(yolo_line)

def convert_class_to_id(class_name):
    # Пример: создаём маппинг классов
    class_map = {
        "3_24_n40": 0,
        "3_24_n20": 1,
        "3_4_n8": 2,
        "3_4_1": 3,
        "3_27": 4,
        "3_18": 5,
        "3_24_n5": 6,
        "3_24_n30": 7,
        "3_24_n60": 8,
        "3_24_n70": 9,
        "3_24_n50": 10,
        "3_32": 11,
        "2_5": 12,
        "3_1": 13,
        "3_20": 14,
        "3_13_r4.5": 15,
        "3_2": 16,
        "3_24_n80": 17,
        "3_10": 18,
        "3_28": 19,
        "3_24_n10": 20,
        "2_6": 21,
        "3_18_2": 22,
        "3_19": 23,
        "3_30": 24,
        "3_29": 25,
        "3_11_n5": 26,
        "3_13_r3.5": 27
    }
    return class_map.get(class_name, -1)  # Возвращаем -1, если класс не найден

# Пример вызова для тренировочных и тестовых данных
convert_csv_to_yolo(
    csv_path=r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\dataset\prohibitory\train_gt.csv",
    images_dir=r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\dataset\frames\train",
    labels_dir=r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\dataset\frames\train\labels"
)

convert_csv_to_yolo(
    csv_path=r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\dataset\prohibitory\test_gt.csv",
    images_dir=r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\dataset\frames\test",
    labels_dir=r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\dataset\frames\test\labels"
)

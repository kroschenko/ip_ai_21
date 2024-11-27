import os
import shutil
import pandas as pd

# Пути к CSV и изображениям
train_csv_path = 'train_gt.csv'
test_csv_path = 'test_gt.csv'
train_images_dir = 'train'
test_images_dir = 'test'

# Папка для структуры YOLO
output_base_dir = 'YOLO_dataset'
train_output_dir = os.path.join(output_base_dir, 'train')
test_output_dir = os.path.join(output_base_dir, 'test')

# Функция для создания структуры YOLO
def create_yolo_structure(csv_path, images_dir, output_dir):
    labels_output_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')

    # Создаем папки
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # Читаем CSV
    data = pd.read_csv(csv_path)

    for _, row in data.iterrows():
        filename = row['filename'].strip()  # Убираем лишние пробелы
        class_id = int(row['sign_class'].split('_')[1])  # Пример: "5_16"
        x_from, y_from, width, height = row['x_from'], row['y_from'], row['width'], row['height']

        # Полный путь к изображению
        img_path = os.path.join(images_dir, filename)

        if not os.path.exists(img_path):
            print(f"Файл не найден: {img_path}, пропускаем.")
            continue

        # Копируем изображение
        shutil.copy(img_path, images_output_dir)

        # Размеры изображения
        from PIL import Image
        img = Image.open(img_path)
        img_width, img_height = img.size

        # YOLO-аннотации
        x_center = (x_from + width / 2) / img_width
        y_center = (y_from + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # Сохраняем аннотацию
        label_file = os.path.join(labels_output_dir, os.path.splitext(filename)[0] + '.txt')
        with open(label_file, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

# Создаем структуру для train и test
create_yolo_structure(train_csv_path, train_images_dir, train_output_dir)
create_yolo_structure(test_csv_path, test_images_dir, test_output_dir)

print(f"Структура YOLO создана в папке: {output_base_dir}")

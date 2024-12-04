from ultralytics import YOLO

def train_model():
    yaml_file = r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\data.yaml"
    save_dir = r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\yolov8_model"
    save_model = r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\yolov8_model\yolov8_prohibitory_signs5\weights\last.pt"

    model = YOLO(save_model)

    model.train(
        data=yaml_file, 
        imgsz=680,
        batch=8,      
        epochs=30,      
        save=True,     
        project=save_dir, 
        name="yolov8_prohibitory_signs",
    )

    print("Обучение завершено. Модель сохранена в", save_dir)

if __name__ == '__main__':
    train_model()

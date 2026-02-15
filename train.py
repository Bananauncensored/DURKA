from ultralytics import YOLO
import os

# ====== Настройка ======
# Путь к wagon.yaml
data_yaml = "wagon.yaml"

# Проверка наличия папок
for folder in ["dataset/images/train", "dataset/images/val"]:
    if not os.path.exists(folder):
        raise Exception(f"Папка {folder} не найдена. Проверь структуру датасета!")

# ====== Создание модели ======
# Берем маленькую модель yolov8n для CPU
model = YOLO("yolov8n.pt")

# ====== Обучение ======
model.train(
    data=data_yaml,  # YAML с train/val
    epochs=50,       # количество эпох (можно меньше для теста)
    imgsz=640,       # размер изображения
    batch=4,         # маленький батч для CPU
    device='cpu',    # обучение на CPU
    name='base_wagon_model'  # имя папки с результатами
)

print("✅ Обучение завершено. Результаты в папке 'runs/train/base_wagon_model'")

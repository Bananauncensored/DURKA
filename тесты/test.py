import os
import shutil
import json
from ultralytics import YOLO

# ========= НАСТРОЙКИ =========
MODEL_PATH = r"runs/detect/base_wagon_model2/weights/best.pt"

INPUT_IMAGES_DIR = r"C:\Users\ilua2\OneDrive\Рабочий_стол\ХЛАМ\Программирование\Хакатон май\5_hakaton\59cc0da6-7d73-4b16-a562-8f2cfcb13c43\cam_1\59cc0da6-7d73-4b16-a562-8f2cfcb13c43"

OUTPUT_IMAGE_DIR = "IMAGE"
REPORT_DIR = "report"
REPORT_PATH = os.path.join(REPORT_DIR, "report.json")

# =============================


def main():
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)

    image_files = sorted([
        f for f in os.listdir(INPUT_IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    current_wagon_id = 1
    wagon_frame_counter = {}
    current_wagon_dir = None

    for filename in image_files:
        image_path = os.path.join(INPUT_IMAGES_DIR, filename)

        results = model.predict(image_path, verbose=False)
        boxes = results[0].boxes if results and results[0].boxes is not None else []
        box_count = len(boxes)

        # ❌ нет вагонов — пропускаем
        if box_count == 0:
            continue

        # ⚠️ два и более — сигнал смены вагона
        if box_count > 1:
            current_wagon_id += 1
            current_wagon_dir = None
            continue

        # ✅ ровно один вагон
        if current_wagon_dir is None:
            current_wagon_dir = os.path.join(
                OUTPUT_IMAGE_DIR, f"wagon_{current_wagon_id}"
            )
            os.makedirs(current_wagon_dir, exist_ok=True)
            wagon_frame_counter[current_wagon_id] = 0

        shutil.copy2(
            image_path,
            os.path.join(current_wagon_dir, filename)
        )

        wagon_frame_counter[current_wagon_id] += 1

    # ---------- ОТЧЁТ ----------
    report = {
        "total_wagons": len(wagon_frame_counter),
        "wagons": []
    }

    for wagon_id, frame_count in wagon_frame_counter.items():
        report["wagons"].append({
            "wagon_id": wagon_id,
            "frames_count": frame_count
        })

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Готово.")
    print(f"Вагонов найдено: {report['total_wagons']}")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()

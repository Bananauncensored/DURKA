import torch
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import shutil
import json

# -------------------
# CONFIG
# -------------------
BASE_DIR = Path(__file__).parent  # папка, где лежит test.py
INPUT_DIR = BASE_DIR / "input_image"
OUTPUT_DIR = BASE_DIR / "output_image"

OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" # я отключил CPU потому что мне лучше ошибка, чем CPU 
IMG_SIZE = 224
MODEL_PATH = "mobilenet_wagon_best.pt"

# -------------------
# TRANSFORMS
# -------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------
# LOAD MODEL
# -------------------
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)) # не забывай weights_only=True, тебе нужны только весы
model.to(DEVICE)
model.eval()

# -------------------
# CLASS NAMES
# -------------------
CLASS_NAMES = ["one_wagon", "transition"]

# -------------------
# PROCESS IMAGES
# -------------------
wagon_count = 0
wagon_photos = {}
current_wagon_photos = 0
current_wagon_paths = []

processed = 0
passed = 0

for img_path in sorted(INPUT_DIR.glob("*.[jp][pn]g")):
    processed += 1

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_class = outputs.argmax(dim=1).item()

    cls = CLASS_NAMES[pred_class]

    # -------- one_wagon --------
    if cls == "one_wagon":
        current_wagon_photos += 1
        current_wagon_paths.append(img_path)

    # -------- transition --------
    elif cls == "transition":
        if current_wagon_photos > 0:
            wagon_count += 1
            wagon_photos[f"wagon_{wagon_count}"] = current_wagon_photos

            # папка вагона
            wagon_dir = OUTPUT_DIR / f"wagon_{wagon_count}"
            wagon_dir.mkdir(parents=True, exist_ok=True)

            # сохраняем: первый, каждый второй, последний
            for i, path in enumerate(current_wagon_paths):
                if i == 0 or i == len(current_wagon_paths) - 1 or i % 2 == 1:
                    shutil.copy(path, wagon_dir / path.name)
                    passed += 1

            # сброс
            current_wagon_photos = 0
            current_wagon_paths = []


# Если последний вагон не закрылся transition
if current_wagon_photos > 0:
    wagon_count += 1
    wagon_photos[f"wagon_{wagon_count}"] = current_wagon_photos

    wagon_dir = OUTPUT_DIR / f"wagon_{wagon_count}"
    wagon_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(current_wagon_paths):
        if i == 0 or i == len(current_wagon_paths) - 1 or i % 2 == 1:
            shutil.copy(path, wagon_dir / path.name)
            passed += 1

# -------------------
# SAVE JSON REPORT
# -------------------
REPORT_PATH = BASE_DIR / "wagon_report.json"
with open(REPORT_PATH, "w") as f:
    json.dump(wagon_photos, f, indent=4)

# -------------------
# PRINT STATS
# -------------------
print(f"Processed {processed} images.")
print(f"Passed filter (one_wagon): {passed} images.")
print(f"Detected wagons: {wagon_count}")
print(f"JSON report saved to {REPORT_PATH}")
print("Input dir exists?", INPUT_DIR.exists())
print("Output dir exists?", OUTPUT_DIR.exists())

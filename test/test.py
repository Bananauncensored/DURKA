import torch
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import shutil
import json

# -------------------
# CONFIG
# -------------------
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_image"
OUTPUT_DIR = BASE_DIR / "output_image"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_WIDTH = 800
IMG_HEIGHT = 450
MODEL_PATH = "mobilenet_wagon_best.pt"

TRANSITION_STREAK = 3      # сколько transition подряд считаем настоящим переходом
MIN_WAGON_FRAMES = 5       # защита от мусорных микро-вагонов

# -------------------
# TRANSFORMS
# -------------------
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.20897625386714935, 0.19960245490074158, 0.18924349546432495],
        std=[0.1252901703119278, 0.12167520821094513, 0.12165164202451706]
    )
])

# -------------------
# LOAD MODEL
# -------------------
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

CLASS_NAMES = ["one_wagon", "transition"]

# -------------------
# STATE
# -------------------
wagon_count = 0
wagon_photos = {}

current_wagon_photos = 0
current_wagon_paths = []

transition_counter = 0

processed = 0
passed = 0

# -------------------
# PROCESS IMAGES
# -------------------
for img_path in sorted(INPUT_DIR.glob("*.[jp][pn]g")):
    processed += 1

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        pred_class = outputs.argmax(dim=1).item()

    cls = CLASS_NAMES[pred_class]

    # ---------- ONE_WAGON ----------
    if cls == "one_wagon":
        transition_counter = 0

        current_wagon_photos += 1
        current_wagon_paths.append(img_path)

    # ---------- TRANSITION ----------
    elif cls == "transition":
        transition_counter += 1

        # пока transition не подтверждён — считаем, что вагон продолжается
        if transition_counter < TRANSITION_STREAK:
            current_wagon_photos += 1
            current_wagon_paths.append(img_path)
            continue

        # transition подтверждён
        if current_wagon_photos >= MIN_WAGON_FRAMES:
            wagon_count += 1
            wagon_photos[f"wagon_{wagon_count}"] = current_wagon_photos

            wagon_dir = OUTPUT_DIR / f"wagon_{wagon_count}"
            wagon_dir.mkdir(parents=True, exist_ok=True)

            for i, path in enumerate(current_wagon_paths):
                if i == 0 or i == len(current_wagon_paths) - 1 or i % 2 == 1:
                    shutil.copy(path, wagon_dir / path.name)
                    passed += 1

        # сброс состояния
        current_wagon_photos = 0
        current_wagon_paths = []
        transition_counter = 0

# -------------------
# LAST WAGON (если не закрылся)
# -------------------
if current_wagon_photos >= MIN_WAGON_FRAMES:
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
print(f"Processed images: {processed}")
print(f"Saved images: {passed}")
print(f"Detected wagons: {wagon_count}")
print(f"JSON report saved to {REPORT_PATH}")
print("Input dir exists?", INPUT_DIR.exists())
print("Output dir exists?", OUTPUT_DIR.exists())
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ------------------
# CONFIG
# ------------------
DATASET_DIR = Path("dataset")
BATCH_SIZE = 16          # CPU-friendly
EPOCHS = 30
LR = 1e-4
IMG_SIZE = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------
# TRANSFORMS
# ------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ------------------
# DATASETS
# ------------------
train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_ds.classes)

# ------------------
# MODEL
# ------------------
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.to(DEVICE)

# ------------------
# OPTIMIZER & LOSS
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# ------------------
# TRAIN LOOP
# ------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ------------------
    # VALIDATION
    # ------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: "
        f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    # SAVE BEST
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mobilenet_wagon_best.pt")

print("Training finished. Best val acc:", best_val_acc)

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
BATCH_SIZE = 8          # CPU-friendly
EPOCHS = 20
LR = 1e-4
IMG_WIDTH = 1280        # Оригинальная ширина для сохранения качества (без downscale)
IMG_HEIGHT = 720        # Оригинальная высота (широкоформатный размер, non-square)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # ------------------
    # CALCULATE CUSTOM MEAN AND STD FROM TRAIN DATASET
    # ------------------
    temp_tf = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])
    temp_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=temp_tf)
    temp_loader = DataLoader(temp_ds, batch_size=32, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0

    for imgs, _ in tqdm(temp_loader, desc="Calculating mean/std"):
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1)  # Flatten по пикселям
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    print(f"Custom mean: {mean.tolist()}")
    print(f"Custom std: {std.tolist()}")

    # ------------------
    # TRANSFORMS WITH CUSTOM MEAN/STD
    # ------------------
    train_tf = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # Масштабируем только если нужно (сохраняет aspect ratio и качество)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),  # Аугментация для разнообразия
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # То же для валидации
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # ------------------
    # DATASETS
    # ------------------
    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # num_workers=4 для ускорения
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
            outputs = model(imgs)  # Модель принимает non-square входы (например, 3x720x1280)
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
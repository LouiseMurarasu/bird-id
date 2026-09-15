import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import birdid

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LEARNING_RATE_BACKBONE = 0.0001
UNFREEZE_FROM = 10
EPOCHS = 30
PATIENCE = 5
EXPERIMENT_NAME = "unfreeze3_earlystop"
RESULTS_DIR = Path("results") / EXPERIMENT_NAME
RESULTS_PATH = RESULTS_DIR / "training_log.csv"
MODEL_PATH = Path("models") / f"{EXPERIMENT_NAME}.pt"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(birdid.IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=birdid.IMAGENET_MEAN, std=birdid.IMAGENET_STD),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=birdid.inference_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = birdid.build_model(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier[3].parameters():
    param.requires_grad = True
    
backbone_params = []
for block in model.features[UNFREEZE_FROM:]:
    for param in block.parameters():
        param.requires_grad = True
        backbone_params.append(param)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {"params": model.classifier[3].parameters(), "lr": LEARNING_RATE},
    {"params": backbone_params, "lr": LEARNING_RATE_BACKBONE},
])

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {trainable:,}\n")

history = []
best_val_loss = float("inf")
epochs_without_improvement = 0
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()

    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)
    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    duration = time.time() - start

    print(f"Epoch {epoch}/{EPOCHS} ({duration:.0f}s)")
    print(f"  train : loss {train_loss:.4f} / acc {train_acc:.2%}")
    print(f"  val   : loss {val_loss:.4f} / acc {val_acc:.2%}")

    history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "duration_s": round(duration),
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("  -> meilleur modèle sauvegardé")
    else:
        epochs_without_improvement += 1
        print(f"  -> pas d'amélioration ({epochs_without_improvement}/{PATIENCE})")
        if epochs_without_improvement >= PATIENCE:
            print(f"\nArrêt anticipé à l'epoch {epoch}.")
            break

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

print(f"\nRésultats : {RESULTS_PATH}")
print(f"Modèle    : {MODEL_PATH}")
from pathlib import Path

import csv 

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

VAL_DIR = "data/split/val"
BATCH_SIZE = 32
NUM_CLASSES = 10
EXPERIMENT_NAME = "augmentation"
MODEL_PATH = Path("models") / f"{EXPERIMENT_NAME}.pt"
MATRIX_PATH = Path("results") / EXPERIMENT_NAME / "confusion_matrix.csv"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
classes = val_dataset.classes

model = models.mobilenet_v3_small()
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int32)

with torch.no_grad():
    for images, labels in val_loader:
        predictions = model(images).argmax(dim=1)
        for true_label, predicted_label in zip(labels, predictions):
            matrix[true_label][predicted_label] += 1

print("Matrice de confusion (lignes = vraie espèce, colonnes = prédiction)\n")
header = "".join(f"{i:>6}" for i in range(NUM_CLASSES))
print(f"{'':<24}{header}")
for i, name in enumerate(classes):
    row = "".join(f"{value:>6}" for value in matrix[i].tolist())
    print(f"{i} {name:<22}{row}")

print("\nPrécision par espèce :")
for i, name in enumerate(classes):
    correct = matrix[i][i].item()
    total = matrix[i].sum().item()
    print(f"  {name:<24} {correct:>3}/{total:<3}  {correct / total:.1%}")

overall = matrix.diagonal().sum().item() / matrix.sum().item()
print(f"\nPrécision globale : {overall:.2%}")

MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MATRIX_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true_class"] + classes)
    for i, name in enumerate(classes):
        writer.writerow([name] + matrix[i].tolist())

print(f"Matrice enregistrée : {MATRIX_PATH}")
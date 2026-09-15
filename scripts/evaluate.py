import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import birdid

VAL_DIR = "data/split/val"
BATCH_SIZE = 32
EXPERIMENT_NAME = "unfreeze3_earlystop"
MATRIX_PATH = Path("results") / EXPERIMENT_NAME / "confusion_matrix.csv"
NUM_CLASSES = birdid.NUM_CLASSES

val_dataset = datasets.ImageFolder(VAL_DIR, transform=birdid.inference_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
classes = val_dataset.classes

model = birdid.load_model(EXPERIMENT_NAME)

matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int32)
top3_correct = 0
top3_total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        for true_label, predicted_label in zip(labels, predictions):
            matrix[true_label][predicted_label] += 1

        top3 = outputs.topk(3, dim=1).indices
        top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
        top3_total += labels.size(0)

print("Matrice de confusion (lignes = vraie espèce, colonnes = prédiction)\n")
header = "".join(f"{i:>6}" for i in range(NUM_CLASSES))
print(f"{'':<24}{header}")
for i, name in enumerate(classes):
    row = "".join(f"{value:>6}" for value in matrix[i].tolist())
    print(f"{i} {name:<22}{row}")

print("\nPrécision par espèce :")
for i, name in enumerate(classes):
    correct = matrix[i][i].item()
    species_total = matrix[i].sum().item()
    print(f"  {name:<24} {correct:>3}/{species_total:<3}  {correct / species_total:.1%}")

overall = matrix.diagonal().sum().item() / matrix.sum().item()
print(f"\nPrécision globale : {overall:.2%}")
print(f"Précision top-3   : {top3_correct / top3_total:.2%}")

MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MATRIX_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["true_class"] + classes)
    for i, name in enumerate(classes):
        writer.writerow([name] + matrix[i].tolist())

print(f"Matrice enregistrée : {MATRIX_PATH}")
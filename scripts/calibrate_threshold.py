import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

import birdid

VAL_DIR = "data/split/val"
BATCH_SIZE = 32
EXPERIMENT_NAME = "unfreeze3_earlystop"

val_dataset = datasets.ImageFolder(VAL_DIR, transform=birdid.inference_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = birdid.load_model(EXPERIMENT_NAME)

confidences = []
correctness = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        batch_conf, batch_pred = probabilities.max(dim=1)

        confidences.extend(batch_conf.tolist())
        correctness.extend((batch_pred == labels).tolist())

correct_conf = [c for c, ok in zip(confidences, correctness) if ok]
wrong_conf = [c for c, ok in zip(confidences, correctness) if not ok]

print(f"Prédictions correctes : {len(correct_conf)}")
print(f"  confiance moyenne : {sum(correct_conf) / len(correct_conf):.1%}")
print(f"Prédictions fausses  : {len(wrong_conf)}")
print(f"  confiance moyenne : {sum(wrong_conf) / len(wrong_conf):.1%}")

print("\nEffet du seuil :")
print(f"{'seuil':>6}{'répondues':>11}{'% répondues':>13}{'précision':>11}")
for threshold in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    kept = [ok for c, ok in zip(confidences, correctness) if c >= threshold]
    if not kept:
        continue
    coverage = len(kept) / len(confidences)
    accuracy = sum(kept) / len(kept)
    print(f"{threshold:>6.2f}{len(kept):>11}{coverage:>12.1%}{accuracy:>11.1%}")
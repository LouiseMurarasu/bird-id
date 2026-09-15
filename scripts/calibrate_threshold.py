from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

EXPERIMENT_NAME = "unfreeze3_earlystop"
MODEL_PATH = Path("models") / f"{EXPERIMENT_NAME}.pt"
VAL_DIR = "data/split/val"
BATCH_SIZE = 32
NUM_CLASSES = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.mobilenet_v3_small()
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

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
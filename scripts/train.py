import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Batchs d'entraînement : {len(train_loader)}")
print(f"Batchs de validation  : {len(val_loader)}")

images, labels = next(iter(train_loader))
print(f"\nForme d'un batch d'images : {tuple(images.shape)}")
print(f"Forme des étiquettes      : {tuple(labels.shape)}")
print(f"Étiquettes du batch       : {labels.tolist()}")

from torchvision import models

NUM_CLASSES = 10
LEARNING_RATE = 0.001

weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=LEARNING_RATE)

model.eval()
with torch.no_grad():
    outputs = model(images)
    loss = criterion(outputs, labels)

print(f"\nForme des prédictions : {tuple(outputs.shape)}")
print(f"Loss initiale : {loss.item():.4f}")
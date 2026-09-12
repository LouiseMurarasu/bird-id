from torchvision import datasets, transforms

TRAIN_DIR = "data/split/train"
VAL_DIR = "data/split/val"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

print(f"Images d'entraînement : {len(train_dataset)}")
print(f"Images de validation  : {len(val_dataset)}")
print(f"Nombre de classes     : {len(train_dataset.classes)}")

print("\nCorrespondance classe / numéro :")
for name, index in train_dataset.class_to_idx.items():
    print(f"  {index} : {name}")

image, label = train_dataset[0]
print(f"\nPremière image  : forme {tuple(image.shape)}, classe {label}")
print(f"Valeurs des pixels : min {image.min():.2f} / max {image.max():.2f}")
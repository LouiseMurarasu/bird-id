import torch
from torchvision import models

NUM_CLASSES = 10

weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

for param in model.parameters():
    param.requires_grad = False

in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)

print("Structure du classifieur :")
print(model.classifier)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParamètres totaux      : {total_params:,}")
print(f"Paramètres entraînables : {trainable_params:,}")

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"\nForme de la sortie : {tuple(output.shape)}")
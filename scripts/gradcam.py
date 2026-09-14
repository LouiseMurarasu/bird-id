import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

EXPERIMENT_NAME = "unfreeze3_earlystop"
MODEL_PATH = Path("models") / f"{EXPERIMENT_NAME}.pt"
VAL_DIR = Path("data/split/val")
OUTPUT_DIR = Path("results") / EXPERIMENT_NAME / "gradcam"
NUM_CLASSES = 10

CLASSES = sorted(p.name for p in VAL_DIR.iterdir() if p.is_dir())

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.mobilenet_v3_small()
in_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

target_layer = model.features[-1]
activations = {}
gradients = {}


def forward_hook(module, inputs, output):
    activations["value"] = output


def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0]


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


def make_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    output = model(tensor)
    predicted = output.argmax(dim=1).item()
    confidence = F.softmax(output, dim=1)[0, predicted].item()

    model.zero_grad()
    output[0, predicted].backward()

    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations["value"]).sum(dim=1).squeeze()
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam = cam.detach().numpy()
    return image.resize((224, 224)), cam, predicted, confidence


image_paths = sys.argv[1:]
if not image_paths:
    print("Usage : python scripts/gradcam.py <chemin_image> [...]")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for image_path in image_paths:
    path = Path(image_path)
    image, cam, predicted, confidence = make_gradcam(path)
    true_class = path.parent.name

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Vraie classe : {true_class}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(cam, cmap="jet", alpha=0.5, extent=(0, 224, 224, 0))
    axes[1].set_title(f"Prédit : {CLASSES[predicted]} ({confidence:.0%})", fontsize=9)
    axes[1].axis("off")

    output_path = OUTPUT_DIR / f"{true_class}_{path.stem}.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=100)
    plt.close(figure)
    print(f"Enregistré : {output_path}")
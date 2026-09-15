import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

import birdid

EXPERIMENT_NAME = "unfreeze3_earlystop"
OUTPUT_DIR = Path("results") / EXPERIMENT_NAME / "gradcam"

CLASSES = birdid.class_names()
transform = birdid.inference_transform

model = birdid.load_model(EXPERIMENT_NAME)

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
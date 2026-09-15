"""Fonctions partagées entre les scripts et l'application."""

import json
from pathlib import Path

import torch
from torchvision import models, transforms

SPECIES_PATH = Path("config/species.json")
IUCN_STATUS_PATH = Path("config/iucn_status.json")
IUCN_CATEGORIES_PATH = Path("config/iucn_categories.json")
MODELS_DIR = Path("models")
NUM_CLASSES = 10
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_species():
    """Retourne les espèces indexées par nom de dossier."""
    with open(SPECIES_PATH, encoding="utf-8") as f:
        species_list = json.load(f)
    return {s["folder"]: s for s in species_list}


def class_names():
    """Retourne les noms de classes dans l'ordre utilisé par le modèle."""
    return sorted(load_species().keys())


def build_model(pretrained=False):
    """Construit MobileNetV3-Small avec une tête à NUM_CLASSES sorties.

    pretrained=True charge les poids ImageNet dans le corps du réseau.
    """
    if pretrained:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small()
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)
    return model


def load_model(experiment_name):
    """Construit le modèle et y charge les poids d'une expérience."""
    model = build_model()
    weights_path = MODELS_DIR / f"{experiment_name}.pt"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def load_iucn_status():
    """Retourne les statuts UICN indexés par nom scientifique."""
    with open(IUCN_STATUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_iucn_categories():
    """Retourne les libellés et descriptions des catégories UICN."""
    with open(IUCN_CATEGORIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def iucn_for(scientific_name):
    """Retourne le statut UICN complet d'une espèce, ou None si absent.

    Le dictionnaire retourné combine le statut de l'espèce (category,
    assessed, trend) et la description de sa catégorie (label, color,
    description).
    """
    data = load_iucn_status()
    entry = data["species"].get(scientific_name)
    if entry is None:
        return None
    categories = load_iucn_categories()
    category = categories.get(entry["category"], {})
    return {**entry, **category}
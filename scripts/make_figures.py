import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib.pyplot as plt
import numpy as np

import birdid

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
REFERENCE = "unfreeze3_earlystop"

EXPERIMENTS = [
    ("baseline", "Baseline\n(tête seule)"),
    ("augmentation", "+ Augmentation"),
    ("unfreeze3", "+ Dégel 3 blocs"),
    ("unfreeze3_earlystop", "+ Early stopping"),
]


def read_log(experiment):
    path = RESULTS_DIR / experiment / "training_log.csv"
    with open(path, encoding="utf-8") as f:
        return [
            {k: float(v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def figure_training_curves():
    rows = read_log(REFERENCE)
    epochs = [r["epoch"] for r in rows]
    val_losses = [r["val_loss"] for r in rows]
    best_epoch = epochs[val_losses.index(min(val_losses))]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="Entraînement")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [r["train_acc"] * 100 for r in rows], label="Entraînement")
    axes[1].plot(epochs, [r["val_acc"] * 100 for r in rows], label="Validation")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    figure.suptitle(
        f"Entraînement du modèle de référence — modèle conservé : epoch {best_epoch:.0f}"
    )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "training_curves.png", dpi=120)
    plt.close(figure)


def figure_experiment_comparison():
    names = []
    accuracies = []
    for experiment, label in EXPERIMENTS:
        rows = read_log(experiment)
        val_losses = [r["val_loss"] for r in rows]
        best = rows[val_losses.index(min(val_losses))]
        names.append(label)
        accuracies.append(best["val_acc"] * 100)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(names, accuracies, color="#4C72B0")
    axis.axhline(10, color="gray", linestyle="--", linewidth=1)
    axis.text(-0.45, 11, "Hasard (10 %)", fontsize=8, color="gray")
    axis.set_ylabel("Accuracy de validation (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Effet de chaque amélioration")

    for bar, value in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f} %",
            ha="center",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "experiment_comparison.png", dpi=120)
    plt.close(figure)


def figure_confusion_matrix():
    path = RESULTS_DIR / REFERENCE / "confusion_matrix.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        matrix = np.array([[int(v) for v in row[1:]] for row in reader])

    species = birdid.load_species()
    display = [species[name]["common_name_fr"] for name in labels]

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")

    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(display, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(display, fontsize=8)
    axis.set_xlabel("Prédiction")
    axis.set_ylabel("Espèce réelle")
    axis.set_title("Matrice de confusion (800 images de validation)")

    threshold = matrix.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            if value == 0:
                continue
            axis.text(
                j, i, value,
                ha="center", va="center", fontsize=8,
                color="white" if value > threshold else "black",
            )

    figure.colorbar(image, shrink=0.8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(figure)


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
figure_training_curves()
figure_experiment_comparison()
figure_confusion_matrix()
print(f"Figures enregistrées dans {FIGURES_DIR}")
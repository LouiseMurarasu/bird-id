import random
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
SPLIT_DIR = Path("data/split")
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

for species_dir in sorted(RAW_DIR.iterdir()):
    if not species_dir.is_dir():
        continue

    images = sorted(species_dir.glob("*.jpg"))
    random.shuffle(images)

    n_val = int(len(images) * VAL_RATIO)
    val_images = images[:n_val]
    train_images = images[n_val:]

    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        target_dir = SPLIT_DIR / split_name / species_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in split_images:
            shutil.copy2(image_path, target_dir / image_path.name)

    print(f"{species_dir.name} : {len(train_images)} train / {len(val_images)} val")

print("\nDécoupage terminé.")
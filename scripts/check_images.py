from pathlib import Path

from PIL import Image

raw_dir = Path("data/raw")
corrupted = []

for species_dir in sorted(raw_dir.iterdir()):
    if not species_dir.is_dir():
        continue

    sizes = []
    for image_path in species_dir.glob("*.jpg"):
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                sizes.append(img.size)
        except Exception as error:
            corrupted.append(image_path)
            print(f"CORROMPUE : {image_path} ({error})")

    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]
    print(f"{species_dir.name} : {len(sizes)} images valides")
    print(f"  largeur  min {min(widths)} / max {max(widths)}")
    print(f"  hauteur  min {min(heights)} / max {max(heights)}")

print(f"\nTotal images corrompues : {len(corrupted)}")
import csv
import json
import time
from pathlib import Path

import requests

API_URL = "https://api.inaturalist.org/v1"
PHOTOS_PER_SPECIES = 5

with open("config/species.json", encoding="utf-8") as f:
    species_list = json.load(f)

attributions = []

for species in species_list:
    print(f"--- {species['scientific_name']} ---")

    params = {
        "taxon_id": species["taxon_id"],
        "quality_grade": "research",
        "photos": "true",
        "photo_license": "cc0,cc-by,cc-by-nc",
        "per_page": PHOTOS_PER_SPECIES,
    }
    response = requests.get(f"{API_URL}/observations", params=params, timeout=30)
    response.raise_for_status()
    observations = response.json()["results"]
    time.sleep(1)

    output_dir = Path("data/raw") / species["folder"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for observation in observations:
        photo = observation["photos"][0]
        url = photo["url"].replace("square", "medium")

        image = requests.get(url, timeout=30)
        image.raise_for_status()

        file_path = output_dir / f"{photo['id']}.jpg"
        file_path.write_bytes(image.content)
        print(f"Téléchargée : {file_path}")

        attributions.append({
            "folder": species["folder"],
            "photo_id": photo["id"],
            "license": photo["license_code"],
            "attribution": photo["attribution"],
            "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
        })

        time.sleep(1)

with open("data/raw/attributions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["folder", "photo_id", "license", "attribution", "observation_url"])
    writer.writeheader()
    writer.writerows(attributions)

print(f"{len(attributions)} crédits enregistrés dans data/raw/attributions.csv")
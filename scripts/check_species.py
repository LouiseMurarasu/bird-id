import json
import time

import requests

API_URL = "https://api.inaturalist.org/v1"
MIN_PHOTOS = 300

with open("config/species.json", encoding="utf-8") as f:
    species_list = json.load(f)

for species in species_list:
    params = {
        "taxon_id": species["taxon_id"],
        "quality_grade": "research",
        "photos": "true",
        "photo_license": "cc0,cc-by,cc-by-nc",
        "per_page": 0,
    }
    response = requests.get(f"{API_URL}/observations", params=params, timeout=30)
    response.raise_for_status()
    total = response.json()["total_results"]

    name = species["scientific_name"]
    if total < MIN_PHOTOS:
        print(f"{name} : {total} photos (ATTENTION : moins de {MIN_PHOTOS})")
    else:
        print(f"{name} : {total} photos")

    time.sleep(1)
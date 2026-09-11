import csv
import json
import time
from pathlib import Path

import requests

API_URL = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "bird-id-portfolio"}
PHOTOS_PER_SPECIES = 5
PER_PAGE = 3
MAX_ATTEMPTS = 3


def get_with_retry(url, params=None):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 5 * attempt
            print(f"Erreur réseau, nouvelle tentative dans {wait} secondes ({attempt}/{MAX_ATTEMPTS})")
            time.sleep(wait)


with open("config/species.json", encoding="utf-8") as f:
    species_list = json.load(f)

attributions = []
seen_photo_ids = set()

for species in species_list:
    print(f"--- {species['scientific_name']} ---")

    observations = []
    page = 1
    while len(observations) < PHOTOS_PER_SPECIES:
        params = {
            "taxon_id": species["taxon_id"],
            "quality_grade": "research",
            "photos": "true",
            "photo_license": "cc0,cc-by,cc-by-nc",
            "per_page": PER_PAGE,
            "page": page,
            "order_by": "id",
            "order": "asc",
        }
        response = get_with_retry(f"{API_URL}/observations", params)
        results = response.json()["results"]
        time.sleep(1)

        print(f"Page {page} : {len(results)} observations")
        if not results:
            break

        for observation in results:
            if len(observations) == PHOTOS_PER_SPECIES:
                break
            photo_id = observation["photos"][0]["id"]
            if photo_id in seen_photo_ids:
                print(f"Doublon ignoré : photo {photo_id}")
                continue
            seen_photo_ids.add(photo_id)
            observations.append(observation)

        page += 1

    output_dir = Path("data/raw") / species["folder"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for observation in observations:
        photo = observation["photos"][0]
        url = photo["url"].replace("square", "medium")

        image = get_with_retry(url)

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
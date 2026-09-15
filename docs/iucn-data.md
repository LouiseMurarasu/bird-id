# Données de conservation UICN

## Choix : données statiques plutôt qu'appel API

L'IUCN propose une API (Red List API v4) donnant accès aux évaluations. Elle n'a
pas été utilisée pour ce projet.

**Raison.** Les conditions d'utilisation de l'API précisent qu'elle est destinée
avant tout aux efforts de conservation, en particulier dans les domaines de
l'éducation et de la recherche, et que l'accès peut être restreint pour des
usages tels que le développement d'applications mobiles, l'inclusion dans des
cours d'informatique ou des projets de visualisation sans lien avec la
conservation.

Ce projet est un exercice de portfolio : son intention première est
l'apprentissage et la démonstration technique, même si son sujet porte sur la
conservation. Demander un jeton pour cet usage n'aurait pas correspondu au
périmètre annoncé par l'IUCN.

**Solution retenue.** Les statuts des 10 espèces ont été relevés manuellement sur
le site public iucnredlist.org et stockés dans `config/iucn_status.json`, avec
citation de la source, du périmètre et de la date de consultation.

**Bénéfices secondaires.** L'application fonctionne sans clé, sans connexion
réseau et sans dépendance externe au moment de la prédiction. Quiconque clone le
dépôt peut la lancer immédiatement. Cela évite également un appel réseau par
photo analysée, ce qui est cohérent avec l'approche Green Coding du projet.

## Source

IUCN 2026. IUCN Red List of Threatened Species. Version 2026-1.
www.iucnredlist.org

Périmètre : évaluations mondiales. Consulté le 15 septembre 2026.

## Statuts relevés

| Espèce | Statut | Évaluation | Tendance |
|---|---|---|---|
| Erithacus rubecula | LC | 2018 | En augmentation |
| Upupa epops | LC | 2020 | En déclin |
| Streptopelia decaocto | LC | 2019 | En augmentation |
| Streptopelia turtur | VU | 2019 | En déclin |
| Vanellus vanellus | NT | 2025 | En déclin |
| Numenius arquata | NT | 2017 | En déclin |
| Aythya ferina | VU | 2021 | En déclin |
| Fratercula arctica | VU | 2018 | En déclin |
| Neophron percnopterus | EN | 2021 | En déclin |
| Oxyura leucocephala | EN | 2017 | En déclin |

**Neuf espèces sur dix ont une population en déclin.** Seuls le rouge-gorge et la
tourterelle turque progressent.

## Limites connues

**Données figées.** Les statuts sont ceux de la version 2026-1. L'UICN réévalue
les espèces régulièrement : le fichier devra être mis à jour à chaque nouvelle
version de la Red List.

**Périmètre mondial uniquement.** Une espèce peut avoir un statut mondial et un
statut régional différents. Le statut européen ou français n'est pas affiché.
Par exemple, une espèce classée LC au niveau mondial peut être menacée en France.

**Affichage conditionnel.** Le statut n'apparaît que si la confiance du modèle
dépasse le seuil. Annoncer un statut de conservation sur une identification
incertaine serait trompeur.
# Découpage du dataset

## Choix retenu

Découpage simple (*holdout*) en deux jeux, stratifié par espèce :

- entraînement : 80 %, soit 320 images par espèce (3200 au total)
- validation : 20 %, soit 80 images par espèce (800 au total)

Graine aléatoire fixée à 42, tirage vérifié comme reproductible.
Les images sont copiées depuis `data/raw`, qui reste intact.

## Pourquoi ce découpage

**Stratifié par espèce.** Le prélèvement de 20 % est fait espèce par espèce, et
non sur l'ensemble des images. Sans cela, le hasard pourrait attribuer un nombre
très inégal d'images de validation d'une espèce à l'autre, et fausser la
comparaison des performances entre espèces.

**Mélange avant découpage.** Les images sont triées de la plus ancienne à la plus
récente sur iNaturalist. Sans mélange, le jeu de validation ne contiendrait que
des photos récentes, prises en moyenne avec du meilleur matériel, donc plus
faciles. Les deux jeux ne seraient pas comparables.

**Graine fixe.** Elle garantit un tirage identique à chaque exécution. Sans elle,
un écart de score entre deux entraînements pourrait venir du découpage plutôt que
du modèle.

## Alternatives écartées

**Validation croisée en k blocs.** Plus fiable, car le score ne dépend plus d'un
seul tirage, mais elle impose k entraînements complets. Avec des classes
parfaitement équilibrées (400 images chacune) et 80 images de validation par
espèce, le gain de précision de mesure ne justifie pas de multiplier par k le
temps de calcul et la consommation énergétique. Choix cohérent avec l'approche
Green Coding du projet.

**Jeu de test séparé (train / val / test).** En toute rigueur, la validation sert
à ajuster les réglages et un troisième jeu, jamais consulté, donne le score final.
Ce découpage en trois n'a pas été retenu pour ce MVP, car peu de réglages seront
ajustés. Le risque est connu : à force d'optimiser en regardant le score de
validation, celui-ci devient légèrement optimiste. À reconsidérer si le projet
s'étend.
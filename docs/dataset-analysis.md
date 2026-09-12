# Analyse du dataset

Inspection visuelle manuelle des 4000 images téléchargées (400 par espèce).
Date : 2026-09-12

## Méthode

Comptage approximatif, dossier par dossier, des images jugées peu exploitables
pour l'entraînement. Les comptages sont volontairement prudents : une photo
lointaine mais nette n'a pas été comptée comme problématique.

## Résultats

| Espèce | Images problématiques | Part | Problème dominant |
|---|---|---|---|
| Erithacus rubecula | ~45 | 11 % | Branches et végétation devant l'oiseau |
| Streptopelia decaocto | ~40 | 10 % | Sujet lointain |
| Streptopelia turtur | ~50 | 13 % | Sujet lointain |
| Upupa epops | 80 | ~20 % | Sujet lointain |
| Fratercula arctica | ~80 | 20 % | Sujet lointain, plusieurs individus |
| Oxyura leucocephala | ~75 | 19 % | Sujet lointain, plusieurs individus |
| Aythya ferina | ~80 | 20 % | Sujet lointain, plusieurs individus |
| Numenius arquata | ~150 | 38 % | Sujet lointain, groupes d'individus |
| Vanellus vanellus | ~200 | 50 % | Sujet lointain ou groupes d'individus |
| Neophron percnopterus | ~225 | 56 % | Photographié en vol, donc lointain |

## Biais identifiés

**Le taux de photos exploitables varie fortement selon l'espèce.** Il va de 89 %
pour le rouge-gorge à 44 % pour le vautour percnoptère. Cela s'explique par la
distance d'observation : un rouge-gorge se photographie à quelques mètres dans un
jardin, un vautour à plusieurs centaines de mètres en vol.

**Risque d'apprentissage du décor plutôt que de l'oiseau.** Plusieurs espèces
apparaissent dans un environnement très homogène :

- Neophron percnopterus : presque toujours en vol sur fond de ciel
- Aythya ferina et Oxyura leucocephala : exclusivement sur l'eau

Le modèle pourrait apprendre à reconnaître le fond plutôt que l'espèce.

**Présence fréquente de plusieurs individus** chez Vanellus vanellus,
Numenius arquata, Fratercula arctica, Aythya ferina et Oxyura leucocephala.
Il s'agit toujours d'individus de la même espèce : l'étiquette reste correcte.

## Décision

Aucune image n'a été supprimée à ce stade. Le dataset est conservé tel quel pour
un premier entraînement. Les confusions observées dans la matrice de confusion
seront comparées à ces observations pour décider d'un éventuel nettoyage.
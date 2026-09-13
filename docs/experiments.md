# Journal d'expériences

Une entrée par entraînement. Les résultats détaillés par epoch sont dans
`results/training_log.csv`, la matrice de confusion dans
`results/confusion_matrix.csv`.

## Repère de référence

- Prédiction au hasard sur 10 classes : 10 % d'accuracy, loss = ln(10) = 2,30
- Loss mesurée avant tout entraînement : 2,35 (conforme)

---

## Expérience 1 — Baseline, couche finale seule

Date : 2026-09-13

### Réglages

| Paramètre | Valeur |
|---|---|
| Modèle | MobileNetV3-Small pré-entraîné ImageNet |
| Couches entraînées | couche finale uniquement (10 250 paramètres) |
| Images | 320 train / 80 val par espèce |
| Préparation | Resize 224×224, normalisation ImageNet |
| Augmentation | aucune |
| Batch size | 32 |
| Optimiseur | Adam |
| Learning rate | 0,001 |
| Epochs | 10 |
| Durée | ~22 s par epoch (CPU) |

### Résultats

- Accuracy validation finale : **70,25 %**
- Accuracy entraînement finale : 76,62 %
- Loss validation finale : 0,873

| Espèce | Précision |
|---|---|
| neophron_percnopterus | 81,2 % |
| fratercula_arctica | 80,0 % |
| erithacus_rubecula | 77,5 % |
| oxyura_leucocephala | 71,2 % |
| vanellus_vanellus | 71,2 % |
| streptopelia_decaocto | 68,8 % |
| upupa_epops | 68,8 % |
| aythya_ferina | 63,7 % |
| streptopelia_turtur | 62,5 % |
| numenius_arquata | 57,5 % |

### Analyse

**Plafonnement précoce.** L'accuracy de validation atteint 70 % dès l'epoch 2 et
n'évolue plus ensuite. L'accuracy d'entraînement continue de monter (64 % → 77 %),
creusant un écart de 7 points. Surapprentissage naissant : prolonger cette
configuration ne servirait à rien.

**Trois paires de confusion dominent**, et représentent l'essentiel des erreurs :

| Paire | Erreurs croisées | Contexte commun |
|---|---|---|
| aythya_ferina / oxyura_leucocephala | 34 | Canards, uniquement sur l'eau |
| numenius_arquata / vanellus_vanellus | 28 | Limicoles, souvent photographiés de loin |
| streptopelia_turtur / streptopelia_decaocto | 21 | Même genre, très proches visuellement |

La confusion entre les deux tourterelles est la plus gênante pour l'objectif du
projet : elle revient à confondre une espèce commune (LC) avec une espèce
menacée (VU).

**Contredit l'hypothèse de l'audit visuel.** Neophron percnopterus, l'espèce avec
le plus de photos jugées problématiques (56 %), obtient la meilleure précision
(81,2 %). Explication probable : c'est la seule espèce photographiée en vol sur
fond de ciel, un contexte qu'aucune autre ne partage. Le modèle dispose donc d'un
indice facile — mais peut-être du décor plutôt que de l'oiseau, ce qui serait
fragile sur des photos réelles.

**Conclusion.** La qualité des photos ne prédit pas la performance. Ce qui compte
est la présence d'une espèce visuellement proche dans un contexte similaire. Une
espèce isolée s'en sort malgré des photos médiocres ; deux espèces proches se
pénalisent mutuellement.

### Pistes pour la suite

1. Data augmentation (rotations, recadrages, variations de luminosité) pour
   limiter le surapprentissage
2. Dégeler les derniers blocs du réseau avec un learning rate réduit, pour
   affiner l'extraction de caractéristiques sur les paires difficiles
3. Vérifier si le vautour est reconnu par l'oiseau ou par le fond
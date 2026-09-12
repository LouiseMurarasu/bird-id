# Choix du modèle et coût de calcul

## Modèle retenu

**MobileNetV3-Small**, pré-entraîné sur ImageNet, avec remplacement de la couche
finale de classification.

## Pourquoi MobileNet plutôt qu'un réseau plus lourd

| Modèle | Paramètres (environ) |
|---|---|
| ResNet50 | 25,6 M |
| EfficientNet-B0 | 5,3 M |
| MobileNetV3-Large | 5,5 M |
| MobileNetV3-Small | 2,5 M |

Moins de paramètres signifie moins de calcul à l'entraînement, mais surtout à
chaque prédiction. L'app fera une prédiction à chaque photo envoyée : c'est là que
le coût se répète.

Trois raisons ont motivé la version Small :

- entraînement sur CPU uniquement (pas de GPU disponible sur la machine de développement)
- prédiction rapide dans l'interface Streamlit
- compatibilité avec une éventuelle exécution sur un petit appareil embarqué
  (piste envisagée : caméra de jardin autonome)

Si la précision obtenue est insuffisante, le passage à MobileNetV3-Large est la
première option à tester.

## Effet du fine-tuning sur le coût d'entraînement

Mesures relevées sur le modèle réellement construit dans `scripts/check_model.py`.

| Étape | Paramètres |
|---|---|
| MobileNetV3-Small d'origine (1000 classes ImageNet) | 2 542 856 |
| Après remplacement de la couche finale (10 classes) | 1 528 106 |
| Paramètres réellement entraînés (couche finale seule) | 10 250 |

**Le remplacement de la couche finale retire environ 1 million de paramètres.**
La couche d'origine contenait 1024 × 1000 poids ; la nouvelle n'en contient que
1024 × 10, plus 10 biais.

**Seuls 0,7 % du réseau sont entraînés.** Tout le corps du réseau est gelé
(`requires_grad = False`) : il conserve les caractéristiques visuelles apprises
sur ImageNet et n'est jamais modifié. Seule la couche finale, créée avec des poids
aléatoires, apprend à associer ces caractéristiques aux 10 espèces.

À noter : le gel réduit le coût de **l'apprentissage**, pas celui de la prédiction.
Les 1,5 million de paramètres gelés sont toujours utilisés pour analyser chaque
image ; ils ne sont simplement jamais mis à jour.

## Suite envisagée

Si la précision plafonne, dégeler les derniers blocs du corps du réseau et les
réentraîner avec un learning rate nettement plus faible que celui de la couche
finale, afin de ne pas détruire les connaissances issues d'ImageNet.
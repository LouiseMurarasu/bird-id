# Explicabilité (Grad-CAM)

## Méthode

Grad-CAM produit une carte de chaleur indiquant les zones de l'image qui ont le
plus contribué à la prédiction.

Principe : on capture les activations de la dernière couche convolutive
(`model.features[-1]`), dernier endroit du réseau où l'information reste spatiale.
On calcule les gradients du score de la classe prédite par rapport à ces
activations, ce qui donne l'importance de chaque carte de caractéristiques. On
combine les cartes pondérées par ces importances, on garde les contributions
positives (ReLU) et on normalise.

Les gradients servent ici à **expliquer**, pas à apprendre : aucun poids n'est
modifié.

**Implémentation maison** plutôt que la librairie `pytorch-grad-cam`. Le code fait
une quinzaine de lignes et permet de comprendre les hooks PyTorch et l'accès aux
activations internes. La librairie reste préférable si l'on veut comparer
plusieurs variantes (Grad-CAM++, ScoreCAM, EigenCAM).

## Résultat 1 — Le biais de décor du vautour est réfuté

**Hypothèse testée.** Neophron percnopterus obtenait le meilleur score de la
baseline (81,2 %) malgré 56 % de photos jugées problématiques. Soupçon : le modèle
reconnaîtrait le fond de ciel uniforme plutôt que l'oiseau.

**Observation.** Sur trois images, les zones chaudes suivent systématiquement
l'oiseau ; le fond reste froid.

| Contexte | Zone d'attention | Confiance |
|---|---|---|
| En vol, fond de ciel | Extrémités des ailes et corps | 100 % |
| En vol, fond de ciel | Silhouette complète | 100 % |
| Au sol, fond herbeux | Tête jaune et plumage clair | 98 % |

**Conclusion.** L'hypothèse est réfutée. Le modèle s'appuie sur des critères
adaptés à la pose : extrémités d'ailes contrastées en vol, tête et plumage au sol.
Le contraste entre les rémiges sombres et le corps clair est précisément le
critère utilisé par les ornithologues pour identifier cette espèce en vol.

**Effet secondaire expliqué.** La chute de 5 points observée avec la data
augmentation s'explique probablement par le `RandomResizedCrop` (0,7–1,0), qui peut
couper les extrémités des ailes — c'est-à-dire l'indice principal en vol.

## Résultat 2 — Les confusions Courlis / Vanneau viennent des données

**Contexte.** Numenius arquata est l'espèce la plus faible (60 %), avec 19
confusions vers Vanellus vanellus.

**Observation sur trois images.**

| Image | Zone d'attention | Confiance |
|---|---|---|
| Oiseau net, de face | Corps et plumage | 99 % |
| Oiseau net, de profil | Corps et plumage | 91 % |
| Oiseau minuscule, très loin | **Herbes en bord d'eau, loin de l'oiseau** | 65 % |

**Conclusion.** Quand l'oiseau est visible, le modèle utilise le plumage. Quand il
est trop petit, il se rabat sur le décor. Sur la troisième image, la prédiction est
correcte mais **pour une mauvaise raison** : le modèle ne peut pas avoir identifié
un courlis à partir de brins d'herbe. Il exploite une corrélation entre végétation
de bord d'eau et présence de courlis dans le jeu de données.

C'est très probablement le mécanisme des confusions avec le vanneau : les deux
espèces fréquentent les mêmes milieux, donc le décor ne peut pas les départager.

**Le problème est dans les données, pas dans le modèle.** L'audit visuel avait
relevé 38 % de photos lointaines pour le courlis et 50 % pour le vanneau.

## Piste ouverte : seuil de confiance

La confiance chute nettement sur l'image non exploitable (65 % contre 91 % et
99 %). Un seuil dans l'application permettrait de répondre « photo trop peu nette
pour identifier avec certitude » plutôt que de produire une fausse certitude.

Valeur à calibrer sur l'ensemble du jeu de validation.

## Leçon générale

Les métriques agrégées ne disent pas *pourquoi* un modèle réussit ou échoue.
Le vautour semblait suspect et s'est révélé sain ; le courlis semblait simplement
difficile et s'est révélé dépendant du décor sur les images pauvres. Sans
Grad-CAM, ces deux diagnostics auraient été inversés.
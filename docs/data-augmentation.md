# Data augmentation

## Principe retenu

Une transformation n'est utile que si elle produit une image **plausible dans le
monde réel**. Transformer dans tous les sens ne rend pas le modèle plus robuste :
cela lui fait dépenser de la capacité d'apprentissage sur des situations qu'il ne
rencontrera jamais.

Le critère appliqué à chaque transformation : « une photo réelle d'oiseau
pourrait-elle ressembler à ça ? »

L'augmentation est appliquée **uniquement au jeu d'entraînement**. Le jeu de
validation conserve un simple redimensionnement, sans quoi les scores ne seraient
pas comparables d'une expérience à l'autre.

## Transformations retenues

| Transformation | Paramètres | Justification |
|---|---|---|
| `RandomResizedCrop` | 224, scale 0.7–1.0 | Simule un oiseau décadré ou partiellement visible. Fait varier le fond d'un passage à l'autre, ce qui limite l'apprentissage du décor. |
| `RandomHorizontalFlip` | p = 0.5 | Un oiseau orienté à gauche ou à droite est également plausible. Double virtuellement le dataset sans coût. |
| `RandomRotation` | ±15° | Correspond à un appareil légèrement penché, situation très fréquente. |
| `ColorJitter` | ±20 % luminosité, contraste, saturation | Simule différentes conditions de lumière : contre-jour, temps couvert, heure dorée. |

Le tirage est refait à chaque epoch : le modèle ne voit jamais deux fois
exactement la même image.

## Transformations écartées

**Retournement vertical.** Un oiseau à l'envers n'existe pas. Le modèle
apprendrait une configuration qu'aucune photo réelle ne présentera.

**Rotation de 90° ou plus.** Même raisonnement. L'amplitude compte autant que la
transformation elle-même : ±15° imite un cadrage imparfait, 90° fabrique une
situation impossible.

**Recadrage trop agressif (scale < 0.5).** Risque de produire des images où
l'oiseau est totalement absent, donc mal étiquetées.

## Résultats mesurés

Comparaison baseline (sans augmentation, 10 epochs) et augmentation (20 epochs).

| Mesure | Baseline | Augmentation |
|---|---|---|
| Accuracy validation | 70,25 % | 71,50 % |
| Loss validation | 0,873 | 0,819 |
| Écart train/val | +6,4 pts | +5,5 pts |

Le gain global est modeste, mais l'effet par espèce est beaucoup plus marqué :

| Espèce | Écart |
|---|---|
| streptopelia_turtur | +11,3 pts |
| aythya_ferina | +8,8 pts |
| neophron_percnopterus | −5,0 pts |
| upupa_epops | −3,8 pts |

**L'augmentation aide là où deux espèces sont proches mais distinguables.** Les
tourterelles passent de 21 à 17 erreurs croisées, les canards de 34 à 28. Le
recadrage aléatoire semble forcer le modèle à examiner l'oiseau plutôt que la
scène entière.

**Elle est impuissante là où l'information visuelle manque.** Courlis et Vanneau
restent à 30 erreurs croisées : quand l'oiseau occupe quelques dizaines de pixels,
aucune transformation ne fait apparaître l'information absente.

**La baisse du vautour percnoptère est un indice intéressant.** C'était l'espèce
soupçonnée d'être reconnue par son contexte (ciel uniforme) plutôt que par
l'oiseau. L'augmentation perturbe justement ce contexte, et sa précision chute de
5 points. Hypothèse à confirmer par Grad-CAM.

## Conclusion

L'augmentation a rempli son rôle : l'écart entre entraînement et validation s'est
réduit, la loss de validation s'est améliorée. Mais l'accuracy d'entraînement
plafonne elle aussi (77 %), ce qui indique que le problème n'est plus le
surapprentissage mais un **manque de capacité** : avec 10 250 paramètres
entraînables, la couche finale ne peut pas exploiter davantage des
caractéristiques figées apprises sur ImageNet.

Prochaine étape : dégeler les derniers blocs du réseau. L'augmentation sera
conservée, car elle protégera du surapprentissage quand le nombre de paramètres
entraînables augmentera.
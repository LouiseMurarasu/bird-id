# Seuil de confiance

## Principe

Le modèle produit toujours une réponse parmi les 10 espèces, même lorsque la photo
ne permet pas d'identifier l'oiseau. Un seuil de confiance permet à l'application
de s'abstenir plutôt que de produire une identification fragile.

La confiance est la probabilité de la classe prédite, obtenue par softmax sur les
10 scores de sortie.

## Le modèle sait quand il ne sait pas

Mesures sur les 800 images de validation, modèle `unfreeze3_earlystop`.

| Groupe | Effectif | Confiance moyenne |
|---|---|---|
| Prédictions correctes | 628 | 85,5 % |
| Prédictions fausses | 172 | 58,8 % |

L'écart de 27 points est le résultat déterminant : les erreurs sont en moyenne
nettement moins confiantes que les réussites. Un seuil est donc exploitable. Si les
deux distributions s'étaient recouvertes, le filtrage n'aurait servi à rien.

## Compromis couverture / précision

| Seuil | Images répondues | Couverture | Précision |
|---|---|---|---|
| 0,00 | 800 | 100 % | 78,5 % |
| 0,50 | 696 | 87,0 % | 84,1 % |
| 0,60 | 624 | 78,0 % | 87,5 % |
| **0,70** | **553** | **69,1 %** | **91,3 %** |
| 0,80 | 481 | 60,1 % | 93,8 % |
| 0,90 | 378 | 47,2 % | 98,1 % |
| 0,95 | 313 | 39,1 % | 98,4 % |

## Seuil retenu : 0,70

**Justification.** La précision passe de 78,5 % à 91,3 %, au prix d'une abstention
sur environ une photo sur trois.

Pour une application de conservation, une identification erronée coûte plus cher
qu'une absence de réponse : l'utilisateur peut reprendre une photo, mais il n'a
aucun moyen de savoir qu'un verdict est faux. Une espèce menacée prise pour une
espèce commune induit en erreur sur l'enjeu de conservation.

**Pourquoi pas plus haut.** À 0,90 la précision atteint 98,1 %, mais l'application
refuserait de répondre à plus d'une photo sur deux, ce qui la rendrait frustrante à
l'usage.

**Pourquoi pas plus bas.** À 0,50, le gain n'est que de 5,6 points pour 13 %
d'abstentions : le compromis est moins intéressant.

## Mise en œuvre prévue

- Sous le seuil, l'application indique que la photo ne permet pas une
  identification fiable, et suggère de se rapprocher ou de recadrer
- Le classement des 3 espèces les plus probables est affiché dans tous les cas,
  ce qui reste informatif même en cas d'abstention
- Le seuil sera exposé comme paramètre réglable dans l'interface

## Limite connue

Le seuil est calibré sur le jeu de validation, qui a servi à sélectionner le
modèle. La valeur est donc probablement un peu optimiste. Un jeu de test séparé
serait nécessaire pour une mesure rigoureuse — limite déjà relevée dans
`docs/dataset-split.md`.
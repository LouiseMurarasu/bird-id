# Stratégie d'entraînement

## Dégel progressif du réseau

### Principe

Le corps de MobileNet est initialement gelé : il extrait des caractéristiques
apprises sur ImageNet sans jamais s'adapter aux espèces du projet. Le dégel
autorise les derniers blocs à s'ajuster.

**Pourquoi les derniers blocs.** Les premières couches détectent des éléments
universels (bords, textures, couleurs), réutilisables tels quels. Les dernières
encodent des combinaisons plus spécifiques à ImageNet : ce sont elles qui ont le
plus à gagner à s'adapter à des oiseaux.

**Configuration retenue.** Blocs 10 à 12 sur les 13 que compte `model.features`.
Choix prudent : assez pour laisser le réseau s'adapter, assez peu pour que
l'entraînement reste faisable sur CPU.

| Configuration | Paramètres entraînables |
|---|---|
| Couche finale seule | 10 250 |
| Couche finale + 3 derniers blocs | 654 890 |

### Learning rates différenciés

Entraîner les blocs dégelés au même rythme que la couche finale détruirait les
connaissances issues d'ImageNet — c'est l'**oubli catastrophique**. Deux groupes
de paramètres sont donc définis, chacun avec son propre learning rate :

| Groupe | Learning rate | Justification |
|---|---|---|
| Couche finale | 0,001 | Poids initialisés au hasard : doit tout apprendre |
| Blocs 10–12 | 0,0001 | Poids déjà pertinents : doivent seulement s'ajuster |

Le facteur 10 entre les deux est la convention usuelle.

### Résultat

Accuracy de validation : 71,50 % → 78,50 %. Les paires visuellement proches
progressent le plus (canards, tourterelles), ce qui confirme que le blocage venait
d'un manque de capacité et non du surapprentissage.

### Note Green Coding

Cette configuration multiplie par 64 le nombre de paramètres entraînés. Elle n'a
été lancée qu'après avoir épuisé les options légères (augmentation). Si le gain
avait été négligeable, conserver le modèle léger aurait été le choix rationnel.

---

## Sauvegarde du meilleur modèle et arrêt anticipé

### Le problème observé

Lors de l'expérience 3 (15 epochs, sans arrêt anticipé) :

| Epoch | Train acc | Val acc | Val loss |
|---|---|---|---|
| 6 | 82,8 % | 78,3 % | 0,646 (minimum) |
| 15 | 91,2 % | 78,4 % | 0,703 |

La loss de validation atteint son minimum à l'epoch 6 puis remonte, alors que
l'accuracy reste plate. Le script sauvegardait le **dernier** modèle : le modèle
conservé était donc moins bon que celui obtenu en cours de route.

**Pourquoi l'accuracy ne voit pas le problème.** Le modèle donne à peu près les
mêmes réponses, mais avec une confiance de plus en plus mal placée : il devient
très sûr de lui, y compris quand il se trompe. La cross-entropy pénalise
précisément cela, l'accuracy non. C'est pourquoi les deux métriques sont suivies,
et pourquoi la **loss** sert de critère de décision.

### Les deux mécanismes

**Checkpointing.** À chaque epoch, la loss de validation est comparée au meilleur
score obtenu. Le modèle n'est sauvegardé que s'il s'améliore.

**Early stopping.** Si la loss de validation ne s'améliore plus pendant
`PATIENCE = 5` epochs consécutives, l'entraînement s'arrête.

**Pourquoi une patience de 5 et non 1.** À l'epoch 5 de l'expérience 4, la loss
s'était dégradée — puis l'epoch 6 a battu le record. Un arrêt à la première
mauvaise epoch aurait raté le meilleur modèle. La patience absorbe ces
fluctuations normales.

### Résultat

| | Sans | Avec |
|---|---|---|
| Epochs exécutées | 15 | 11 |
| Modèle conservé | epoch 15 | epoch 6 |
| Val loss du modèle conservé | 0,703 | 0,628 |
| Écart train/val du modèle conservé | 12,8 pts | 4,5 pts |

Meilleur modèle, moins de calcul. L'écart train/val passe de 12,8 à 4,5 points :
un modèle qui généralise remplace un modèle qui avait commencé à mémoriser.
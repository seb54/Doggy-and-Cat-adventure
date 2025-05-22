# Le Tonnetz (Tonal Centroid Features) en analyse musicale

## Introduction

Le **Tonnetz** (réseau tonal en allemand) est une représentation géométrique des relations harmoniques entre les notes de musique. Originaire de la théorie musicale du XIXe siècle, notamment des travaux de Leonhard Euler, ce concept a été adapté au traitement du signal audio sous la forme de descripteurs appelés *Tonal Centroid Features*. 

Ces descripteurs permettent de capturer et de visualiser les relations harmoniques entre les notes d'un morceau musical, offrant ainsi une représentation plus abstraite et musicalement pertinente que les simples hauteurs de notes. L'objectif principal du Tonnetz est de modéliser la proximité perceptive entre différentes tonalités et accords, facilitant ainsi l'analyse des progressions harmoniques.

## Définition intuitive

Le Tonnetz audio peut être compris comme une **projection des données musicales dans un espace à 6 dimensions**. Dans cet espace, chaque dimension représente une relation harmonique fondamentale en musique occidentale:

- Les relations de quintes (Do-Sol, Ré-La, etc.)
- Les relations de tierces majeures (Do-Mi, Ré-Fa#, etc.)
- Les relations de tierces mineures (Do-Mib, Ré-Fa, etc.)

Cette projection transforme le contenu harmonique d'un son en un point dans cet espace multidimensionnel. Deux accords ou tonalités perçus comme proches par l'oreille humaine (par exemple Do majeur et Sol majeur) seront également proches dans cet espace mathématique, ce qui reflète bien notre perception musicale.

## Lien avec les descripteurs chroma

Les features Tonnetz sont directement **dérivées des chromagrammes** (ou vecteurs chroma), qui représentent l'énergie spectrale associée à chacune des 12 classes de hauteur de la gamme chromatique (Do, Do#, Ré, etc.).

La transformation du chroma en Tonnetz s'effectue par une projection mathématique:

1. Le chromagramme capture la présence des 12 notes chromatiques
2. Le Tonnetz transforme cette information en 6 coordonnées représentant des relations harmoniques

Cette projection permet de passer d'une représentation basée sur les hauteurs absolues (chroma) à une représentation basée sur les relations entre ces hauteurs (Tonnetz). Cela offre une information plus structurée sur la tonalité et l'harmonie, faisant ressortir les changements d'accords et les modulations de façon plus évidente.

## Applications courantes

Le Tonnetz trouve de nombreuses applications en analyse musicale computationnelle:

- **Détection de tonalité**: identification de la tonalité principale d'un morceau ou de ses modulations
- **Reconnaissance d'accords**: classification des accords joués grâce à leur empreinte caractéristique dans l'espace Tonnetz
- **Analyse musicale avancée**: étude des progressions harmoniques, identification des cadences et des modulations
- **Classification de morceaux**: regroupement d'œuvres musicales selon leurs caractéristiques harmoniques
- **Comparaison de similitude**: mesure de la proximité harmonique entre différentes pièces musicales
- **Segmentation structurelle**: détection des changements de sections basés sur l'évolution harmonique

## Exemple de code Python

Voici un exemple simple utilisant la bibliothèque `librosa` pour calculer les features Tonnetz à partir d'un fichier audio:

```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Charger un fichier audio
fichier = "morceau.wav"
y, sr = librosa.load(fichier)

# Calculer le spectrogramme à court terme
S = np.abs(librosa.stft(y))

# Calculer le chromagramme
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

# Calculer les features Tonnetz
tonnetz = librosa.feature.tonnetz(y=y, sr=sr, chroma=chroma)

# Afficher le résultat
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagramme')

plt.subplot(2, 1, 2)
librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
plt.colorbar()
plt.title('Tonnetz')

plt.tight_layout()
plt.show()
```

## Visualisation et interprétation

Les données Tonnetz peuvent être visualisées de plusieurs façons:

1. **Série temporelle**: en traçant l'évolution des 6 dimensions au cours du temps, on peut observer les changements harmoniques dans le morceau

2. **Réduction dimensionnelle**: en utilisant des techniques comme la PCA (Analyse en Composantes Principales), on peut projeter les 6 dimensions en 2D ou 3D pour une visualisation plus intuitive

3. **Trajectoire tonale**: en connectant les points successifs dans l'espace Tonnetz, on obtient une trajectoire qui représente l'évolution harmonique du morceau

L'interprétation de ces représentations permet de comprendre la structure harmonique d'une œuvre:

- Un **point stationnaire** dans l'espace Tonnetz indique une harmonie stable
- Un **saut brusque** correspond à un changement harmonique important
- Des **motifs circulaires** peuvent révéler des progressions d'accords répétitives
- Des **zones denses** montrent des sections harmoniquement complexes

## Limites et remarques

Malgré ses avantages, le Tonnetz présente certaines limitations:

- **Sensibilité au bruit**: les signaux bruités peuvent générer des représentations Tonnetz instables
- **Inadéquation aux sons non-harmoniques**: peu efficace pour les percussions ou certaines musiques électroniques
- **Biais occidental**: basé sur la théorie harmonique occidentale, moins pertinent pour d'autres traditions musicales
- **Complexité d'interprétation**: les 6 dimensions peuvent être difficiles à interpréter sans réduction dimensionnelle

Le Tonnetz est particulièrement efficace pour:
- La musique tonale occidentale (classique, pop, rock, jazz)
- Les enregistrements de bonne qualité avec peu de bruit de fond
- L'analyse des progressions d'accords et des modulations

## Conclusion

Le Tonnetz représente une approche puissante pour l'analyse des structures harmoniques dans le signal audio musical. En transformant les données brutes (spectrogramme) en une représentation plus abstraite basée sur les relations harmoniques, il permet de capturer des aspects musicalement pertinents qui échappent aux descripteurs plus simples.

Cette représentation facilite l'analyse automatique de la tonalité, des accords et des progressions harmoniques, offrant ainsi un outil précieux tant pour la recherche musicologique que pour les applications pratiques comme la recherche par similarité ou la génération musicale.

Le Tonnetz illustre parfaitement comment des concepts issus de la théorie musicale traditionnelle peuvent être intégrés dans des méthodes modernes de traitement du signal audio, créant ainsi un pont entre musique et mathématiques. 
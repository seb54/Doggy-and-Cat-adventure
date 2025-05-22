# Le Chroma STFT dans le traitement du signal audio

## Introduction

Le **Chroma STFT** (Short-Time Fourier Transform) est une représentation particulière du signal audio qui met l'accent sur le contenu harmonique en fonction des classes de hauteur musicales. Le terme "chroma" vient du grec *chroma* signifiant "couleur", et fait référence à la "couleur" ou qualité tonale du son. 

Dans le domaine musical, le chroma représente les 12 classes de hauteur de la gamme chromatique occidentale (do, do#, ré, etc.) indépendamment de leur octave. Cette représentation est donc fondamentale pour analyser le contenu harmonique d'un morceau de musique, car elle permet de réduire l'information spectrale complète à une dimension pertinente musicalement.

## Principe de fonctionnement

Le Chroma STFT fonctionne en deux étapes principales :

1. **Application de la STFT (Short-Time Fourier Transform)** : Cette première étape décompose le signal audio en segments courts qui se chevauchent, puis applique une transformée de Fourier à chaque segment. Cela produit une représentation temps-fréquence appelée spectrogramme.

2. **Projection sur les 12 classes chromatiques** : Les fréquences du spectrogramme sont ensuite regroupées en 12 bandes correspondant aux 12 notes de la gamme chromatique (do, do#, ré, etc.). Pour ce faire, toutes les fréquences correspondant à la même note dans différentes octaves sont additionnées.

En termes mathématiques, si nous avons un spectre de fréquences *X(t,f)* à un instant *t*, le chroma *C(t,p)* pour la classe de hauteur *p* (p ∈ {0,1,...,11}) est calculé en sommant les magnitudes spectrales sur toutes les fréquences correspondant à cette classe :

*C(t,p) = Σ X(t,f) pour tout f appartenant à la classe p*

## Utilité de la représentation chroma

Le Chroma STFT présente plusieurs avantages majeurs :

- **Invariance à l'octave** : La même note jouée dans différentes octaves sera représentée par la même classe chromatique, ce qui permet de se concentrer sur le contenu harmonique.

- **Dimension réduite** : Au lieu de traiter des milliers de bandes de fréquence, on se limite à 12 dimensions, ce qui simplifie considérablement l'analyse.

- **Pertinence musicale** : Les 12 classes correspondent directement aux notes musicales, rendant l'interprétation intuitive pour les analyses musicales.

- **Robustesse** : Cette représentation est relativement robuste aux variations de timbre, ce qui permet de comparer des interprétations différentes d'un même morceau.

## Applications courantes

### Détection de la tonalité
Le profil chromatique moyen d'un morceau peut révéler sa tonalité principale en identifiant les notes les plus présentes et leurs relations.

### Reconnaissance d'accords
Les accords musicaux produisent des motifs caractéristiques dans la représentation chroma. Par exemple, un accord de Do majeur activera fortement les bandes Do, Mi et Sol.

### Comparaison de séquences musicales
La similarité entre deux séquences musicales peut être évaluée en comparant leurs représentations chromatiques, ce qui est utile pour identifier des reprises ou des variations d'un même thème.

### Classification de musique
Les caractéristiques chromatiques peuvent servir à catégoriser la musique par genre ou style, car certains genres présentent des progressions harmoniques typiques.

## Exemple de code Python

Voici un exemple simple d'extraction et de visualisation d'un Chroma STFT avec la bibliothèque librosa :

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Chargement du fichier audio
fichier_audio = "exemple_musical.wav"
y, sr = librosa.load(fichier_audio)

# Extraction du Chroma STFT
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Configuration de la figure
plt.figure(figsize=(12, 4))

# Affichage du Chroma
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Représentation Chroma STFT')
plt.tight_layout()
plt.show()

# Afficher la moyenne des valeurs chromatiques (profil tonal global)
chroma_mean = np.mean(chroma, axis=1)
plt.figure(figsize=(8, 4))
plt.bar(range(12), chroma_mean)
plt.xticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.title('Profil chromatique moyen')
plt.xlabel('Note')
plt.ylabel('Magnitude moyenne')
plt.tight_layout()
plt.show()
```

## Visualisation et interprétation

Une représentation Chroma STFT typique se présente sous forme d'une matrice où :
- L'axe horizontal représente le temps
- L'axe vertical représente les 12 classes de hauteur (do à si)
- La couleur indique l'intensité de chaque classe à chaque instant

![Exemple de visualisation Chroma](https://i.imgur.com/exemple_chroma.png)

Pour interpréter cette visualisation :
- Les zones plus claires/chaudes indiquent une forte présence de la note correspondante
- Les motifs verticaux représentent souvent des accords
- Les motifs horizontaux montrent l'évolution harmonique au fil du temps
- Un changement global de motif peut indiquer un changement de section dans le morceau

## Limites et compléments

Le Chroma STFT présente certaines limitations :

- **Insensibilité aux timbres** : Puisqu'il se concentre uniquement sur les hauteurs, il ne capture pas les différences de timbre entre instruments.
- **Inefficacité pour les percussions** : Les sons non harmoniques comme la batterie ne sont pas bien représentés.
- **Sensibilité au bruit** : Le bruit de fond peut affecter la précision de la représentation.

D'autres représentations complémentaires incluent :

- **MFCC (Mel-Frequency Cepstral Coefficients)** : Meilleurs pour capturer le timbre et la texture sonore.
- **Chroma CQT (Constant-Q Transform)** : Une alternative utilisant une transformation à Q constant, offrant une meilleure résolution dans les basses fréquences.
- **Chromagramme Energy Normalized** : Une version normalisée en énergie qui peut améliorer la robustesse aux variations de volume.

## Conclusion

Le Chroma STFT constitue une représentation puissante pour l'analyse musicale, offrant un pont entre le signal audio brut et des concepts musicaux de haut niveau comme les accords et les tonalités. Sa capacité à réduire le signal à ses composantes harmoniques essentielles, tout en restant invariant à l'octave, en fait un outil précieux pour de nombreuses applications d'analyse musicale.

En condensant l'information spectrale complexe en 12 dimensions musicalement pertinentes, le Chroma STFT permet d'aborder des problèmes d'analyse musicale avec une approche plus intuitive et computationnellement efficace que les représentations spectrales complètes.

Pour les chercheurs et développeurs travaillant dans le domaine du MIR (Music Information Retrieval), le Chroma STFT reste un descripteur fondamental, souvent utilisé comme point de départ avant d'explorer des représentations plus spécialisées ou complexes. 
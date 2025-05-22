# Le Spectral Centroid : un descripteur fondamental du timbre sonore

## Introduction

Le **Spectral Centroid** (ou centroïde spectral en français) est l'un des descripteurs les plus importants dans l'analyse du signal audio. Il représente le "centre de gravité" du spectre sonore, indiquant où se concentre l'énergie dans la distribution des fréquences. En termes simples, il nous informe sur la **brillance** perçue d'un son. Cette mesure est fondamentale dans le traitement du signal audio car elle permet de quantifier une caractéristique du timbre qui est facilement perceptible par l'oreille humaine.

## Définition et interprétation

Le Spectral Centroid correspond à la **fréquence moyenne** du signal, pondérée par l'amplitude. En termes perceptifs :

- Un Spectral Centroid **élevé** indique un son *brillant*, *clair* ou *aigu* (riche en hautes fréquences)
- Un Spectral Centroid **bas** indique un son *sourd*, *sombre* ou *rond* (dominé par les basses fréquences)

**Analogie simple** : Imaginez une balançoire avec des poids différents. Le point d'équilibre de cette balançoire se déplacera vers les poids les plus lourds. De la même façon, le Spectral Centroid se déplace vers les fréquences ayant le plus d'énergie dans le spectre sonore.

Exemples concrets :
- Une flûte produit un son avec un centroïde élevé (brillant)
- Un tuba produit un son avec un centroïde bas (sombre)

## Formule mathématique

Le Spectral Centroid se calcule selon la formule suivante :

```
SC = Σ(f[k] × A[k]) / Σ(A[k])
```

Où :
- `f[k]` représente la fréquence de la k-ième composante spectrale
- `A[k]` représente l'amplitude (ou magnitude) de la k-ième composante spectrale
- `Σ` indique la somme de toutes les composantes spectrales

**Exemple intuitif** : Supposons que nous ayons seulement trois fréquences dans notre spectre :
- 100 Hz avec une amplitude de 0.5
- 500 Hz avec une amplitude de 1.0
- 1000 Hz avec une amplitude de 0.2

Le Spectral Centroid serait :
```
SC = (100×0.5 + 500×1.0 + 1000×0.2) / (0.5 + 1.0 + 0.2)
   = (50 + 500 + 200) / 1.7
   = 750 / 1.7
   ≈ 441 Hz
```

Ce résultat indique que le "centre de gravité" spectral se situe autour de 441 Hz.

## Applications courantes

Le Spectral Centroid est utilisé dans de nombreux domaines :

1. **Classification de sons** : Identification automatique d'instruments de musique
2. **Analyse du timbre musical** : Caractérisation objective des timbres
3. **Traitement de la parole** : Distinction entre phonèmes sourds et sonores
4. **Indexation audio** : Recherche de sons similaires dans des bases de données
5. **Égalisation automatique** : Ajustement adaptatif des équaliseurs
6. **Synthèse sonore** : Contrôle de la brillance dans les synthétiseurs

## Exemple de code Python

Voici comment calculer le Spectral Centroid d'un fichier audio avec la bibliothèque `librosa` :

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Charger un fichier audio
y, sr = librosa.load('mon_fichier_audio.wav')

# Calculer le Spectral Centroid
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

# Convertir en secondes pour l'axe des x
frames = range(len(centroid))
t = librosa.frames_to_time(frames, sr=sr)

# Afficher les résultats
print(f"Valeur moyenne du Spectral Centroid : {np.mean(centroid):.2f} Hz")
```

Ce code extrait le Spectral Centroid et affiche sa valeur moyenne. La valeur est exprimée en Hertz (Hz).

## Visualisation

Pour visualiser l'évolution du Spectral Centroid dans le temps, on peut le superposer à un spectrogramme :

```python
# Créer un spectrogramme
plt.figure(figsize=(12, 8))
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.subplot(2, 1, 1)
librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme')

# Afficher le Spectral Centroid
plt.subplot(2, 1, 2)
plt.plot(t, centroid, color='r')
plt.axhline(y=np.mean(centroid), color='k', linestyle='--', 
           label=f'Moyenne: {np.mean(centroid):.0f} Hz')
plt.legend()
plt.xlabel('Temps (s)')
plt.ylabel('Fréquence (Hz)')
plt.title('Spectral Centroid')

plt.tight_layout()
plt.show()
```

Cette visualisation permet d'observer comment la brillance du son évolue au cours du temps. Par exemple, on pourrait voir que :
- Les attaques des notes ont souvent un centroïde plus élevé
- Certains instruments maintiennent un centroïde stable, d'autres varient beaucoup
- Le centroïde peut révéler la structure d'un morceau (couplets, refrains)

## Limites et compléments

Le Spectral Centroid présente certaines limites :

- Il s'agit d'une **mesure unidimensionnelle** qui ne capture qu'un aspect du timbre
- Il peut être **sensible au bruit** de fond dans les enregistrements
- Il ne distingue pas les différentes distributions spectrales pouvant avoir le même centre de gravité

Pour une analyse plus complète, on l'utilise souvent avec d'autres descripteurs spectraux :

- **Spectral Bandwidth** : mesure la "largeur" de la distribution spectrale (l'étalement des fréquences)
- **Spectral Rolloff** : fréquence en dessous de laquelle se trouve un certain pourcentage (généralement 85%) de l'énergie spectrale
- **Spectral Flatness** : indique si le spectre est plat (bruit) ou tonal (notes)
- **Spectral Contrast** : mesure la différence entre pics et vallées dans le spectre
- **MFCC** (Mel-Frequency Cepstral Coefficients) : représentation compacte du timbre

## Conclusion

Le Spectral Centroid est un descripteur audio puissant et intuitif qui quantifie la brillance perçue d'un son. Sa simplicité conceptuelle, combinée à sa pertinence perceptive, en fait un outil fondamental dans l'analyse audio, la classification et le traitement du son. 

Bien que ce ne soit qu'une facette de la complexité du timbre sonore, il fournit une information précieuse et facilement interprétable, ce qui explique sa popularité dans les applications d'analyse et de classification audio. En le combinant avec d'autres descripteurs spectraux, on obtient une caractérisation plus complète des sons, ouvrant la voie à des applications sophistiquées en traitement du signal audio. 
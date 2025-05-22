# Le Spectral Bandwidth en traitement du signal audio

## Introduction

Le **Spectral Bandwidth** (largeur de bande spectrale) est un descripteur audio fondamental qui mesure l'étendue de la distribution des fréquences dans un signal sonore. Ce paramètre permet de quantifier la "largeur" du spectre fréquentiel, offrant ainsi des informations précieuses sur la composition et les caractéristiques du son analysé. En d'autres termes, il indique si l'énergie du signal est concentrée autour de quelques fréquences spécifiques ou dispersée sur une large gamme de fréquences.

## Définition intuitive

Intuitivement, le Spectral Bandwidth mesure la "largeur" du spectre de fréquences autour de son centre de gravité (le Spectral Centroid). Cette mesure nous renseigne directement sur la perception du timbre :

- Un **faible Bandwidth** indique que l'énergie sonore est concentrée autour d'une fréquence centrale, produisant des sons plus "purs" ou "nets" (comme une flûte ou un sifflement).
- Un **Bandwidth élevé** révèle une énergie sonore dispersée sur une large plage de fréquences, caractéristique des sons "riches", "complexes" ou "bruités" (comme une cymbale, un bruit blanc ou un accord de guitare distordue).

La largeur de bande spectrale est donc directement liée à notre perception de la "brillance", de la "chaleur" ou de la "complexité" d'un son.

## Formule mathématique

Mathématiquement, le Spectral Bandwidth est calculé comme l'écart-type des fréquences du signal, pondérées par leur amplitude. Sa formule est la suivante :

```
Bandwidth = √[ Σ((f - Centroid)² × A(f)) / Σ(A(f)) ]
```

Où :
- `f` représente chaque fréquence présente dans le signal
- `Centroid` est le Spectral Centroid (centre de gravité spectral)
- `A(f)` est l'amplitude (ou magnitude) de la fréquence `f`

**Exemple simple** : Considérons un signal qui contient uniquement trois fréquences :
- 100 Hz avec une amplitude de 0.5
- 200 Hz avec une amplitude de 1.0
- 300 Hz avec une amplitude de 0.5

Le Spectral Centroid serait de 200 Hz, et le Bandwidth mesurerait la dispersion autour de cette valeur centrale, pondérée par les amplitudes respectives.

## Lien avec le Spectral Centroid

Le Spectral Bandwidth est intrinsèquement lié au **Spectral Centroid**. Si le Centroid représente la "moyenne" ou le "centre de gravité" des fréquences présentes dans le signal, le Bandwidth mesure leur dispersion autour de ce centre. Ces deux descripteurs fonctionnent donc en tandem :

- Le **Centroid** indique où se situe le "centre" du spectre (sons graves vs aigus)
- Le **Bandwidth** précise comment l'énergie se répartit autour de ce centre (sons concentrés vs diffus)

Un son peut avoir le même Centroid qu'un autre, mais un Bandwidth très différent, ce qui affectera considérablement sa perception.

## Applications courantes

Le Spectral Bandwidth trouve de nombreuses applications dans le traitement et l'analyse audio :

- **Analyse du timbre** : caractérisation des instruments et des voix
- **Classification audio** : identification automatique de genres musicaux ou de types de sons
- **Détection d'instruments** : reconnaissance des instruments dans un mélange sonore
- **Analyse vocale** : différenciation entre parole et chant, ou entre différents locuteurs
- **Détection d'événements acoustiques** : identification de situations sonores particulières
- **Mastering audio** : ajustement de l'équilibre spectral dans les productions musicales

## Exemple de code Python

Voici un exemple simple utilisant la bibliothèque `librosa` pour calculer le Spectral Bandwidth d'un fichier audio :

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Chargement du fichier audio
y, sr = librosa.load('fichier_audio.wav', sr=None)

# Calcul du Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

# Conversion en dB pour une meilleure lisibilité
spectral_bandwidth_db = librosa.amplitude_to_db(spectral_bandwidth)

print(f"Valeur moyenne du Spectral Bandwidth : {np.mean(spectral_bandwidth[0])} Hz")
```

On peut également spécifier différents paramètres, comme la taille de la fenêtre d'analyse ou le type de pondération :

```python
# Avec paramètres personnalisés
spectral_bandwidth = librosa.feature.spectral_bandwidth(
    y=y, 
    sr=sr,
    n_fft=2048,        # Taille de la FFT
    hop_length=512,    # Pas entre les fenêtres d'analyse
    p=2                # Ordre du moment (2 = écart-type standard)
)
```

## Visualisation possible

Il est souvent utile de visualiser l'évolution temporelle du Spectral Bandwidth pour comprendre comment la richesse spectrale d'un son évolue :

```python
import librosa
import matplotlib.pyplot as plt

# Chargement du fichier audio
y, sr = librosa.load('fichier_audio.wav', sr=None)

# Calcul du Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

# Création des points temporels
times = librosa.times_like(spectral_bandwidth, sr=sr, hop_length=512)

# Visualisation
plt.figure(figsize=(12, 4))
plt.plot(times, spectral_bandwidth)
plt.title('Évolution du Spectral Bandwidth')
plt.xlabel('Temps (secondes)')
plt.ylabel('Bandwidth (Hz)')
plt.tight_layout()
plt.show()
```

Cette visualisation permet d'identifier les moments où le son devient plus riche ou plus simple spectralement.

## Limites et compléments

Bien que très utile, le Spectral Bandwidth présente certaines limites :

- Il ne capture qu'un aspect de la distribution spectrale (sa dispersion)
- Il peut être sensible au bruit de fond et aux artefacts de traitement
- Seul, il ne permet pas de distinguer certains types de distributions spectrales

Pour une analyse plus complète, il est généralement combiné avec d'autres descripteurs spectraux :

- **Spectral Centroid** : centre de gravité du spectre
- **Spectral Rolloff** : fréquence en dessous de laquelle se trouve X% de l'énergie spectrale
- **Spectral Flatness** : mesure de l'uniformité du spectre (son tonal vs bruit)
- **Spectral Contrast** : différence entre pics et vallées dans le spectre
- **Mel-Frequency Cepstral Coefficients (MFCCs)** : représentation compacte du spectre adaptée à la perception humaine

## Conclusion

Le Spectral Bandwidth constitue un outil précieux pour quantifier la "largeur" ou la "richesse" du contenu fréquentiel d'un signal audio. En mesurant la dispersion des fréquences autour du centre spectral, il nous renseigne sur des aspects fondamentaux du timbre sonore. Combiné à d'autres descripteurs, il permet de caractériser et classifier efficacement les sons, facilitant ainsi de nombreuses applications en traitement du signal audio, en analyse musicale ou en reconnaissance sonore. 
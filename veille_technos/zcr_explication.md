# Le Zero Crossing Rate (ZCR) en traitement du signal audio

## Introduction

Le Zero Crossing Rate (ZCR), ou taux de passage par zéro en français, est l'une des caractéristiques les plus simples mais aussi les plus utiles dans l'analyse de signaux audio. Il mesure la fréquence à laquelle un signal passe d'une valeur positive à négative (ou inversement) au cours du temps. En d'autres termes, il compte combien de fois le signal "traverse" l'axe horizontal (la valeur zéro).

Cette mesure peut sembler basique, mais elle fournit des informations précieuses sur la nature d'un signal sonore, notamment sur son contenu fréquentiel et sa texture.

## Définition mathématique

Mathématiquement, le ZCR est défini comme la proportion d'échantillons consécutifs qui ont des signes différents. Pour un signal discret, la formule est :

```
ZCR = (1/T-1) * Σ |sign(x[t]) - sign(x[t-1])| / 2
```

Où :
- T est le nombre total d'échantillons
- x[t] représente la valeur du signal à l'instant t
- sign(x) est la fonction signe, qui renvoie +1 si x est positif, -1 si x est négatif

Plus simplement, on peut comprendre le ZCR comme :

*ZCR = (Nombre de passages par zéro) / (Longueur du signal)*

**Exemple :** 
- Un signal sinusoïdal pur à 440 Hz échantillonné à 44100 Hz aura un ZCR proche de 880/44100 = 0,02, car il traverse l'axe horizontal deux fois par période.
- Un bruit blanc aura un ZCR beaucoup plus élevé (proche de 0,5) car il change de signe très fréquemment et de manière aléatoire.

## Interprétation du ZCR

Le ZCR nous renseigne sur plusieurs aspects d'un signal audio :

- **Fréquence** : Plus la fréquence d'un son pur est élevée, plus son ZCR sera élevé.
- **Type de son** : 
  * Les sons voisés (comme les voyelles dans la parole) ont généralement un ZCR bas.
  * Les sons non voisés (comme les consonnes fricatives "s", "f") ont un ZCR élevé.
  * La musique a souvent un ZCR plus stable que la parole.
- **Texture sonore** : Un ZCR élevé peut indiquer un son "bruité" ou "rugueux".

La distribution du ZCR sur différentes fenêtres temporelles peut également révéler la structure d'un signal (silence vs son, parole vs musique, etc.).

## Applications courantes

Le ZCR est utilisé dans de nombreuses applications de traitement audio :

* **Distinction voix/musique** : La parole alterne entre sons voisés (ZCR bas) et non voisés (ZCR élevé), créant une variance du ZCR plus grande que pour la musique.
* **Détection de silence** : Les segments silencieux ont généralement un ZCR très bas.
* **Classification de sons** : Combiné à d'autres caractéristiques, le ZCR aide à différencier les types de sons.
* **Détection de la hauteur tonale** : Pour des sons harmoniques simples, le ZCR peut donner une estimation grossière de la fréquence fondamentale.
* **Segmentation audio** : Les changements dans le ZCR peuvent indiquer des transitions entre différents types de sons.

## Exemple de code Python

Voici comment calculer le ZCR d'un signal audio en Python, en utilisant librosa :

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Charger un fichier audio
y, sr = librosa.load('audio_example.wav', sr=None)

# Calculer le ZCR
zcr = librosa.feature.zero_crossing_rate(y)

# Afficher le ZCR moyen
print(f"ZCR moyen : {np.mean(zcr)}")

# Visualiser le ZCR au cours du temps
plt.figure(figsize=(12, 4))
plt.plot(zcr[0])
plt.title('Zero Crossing Rate')
plt.xlabel('Trames')
plt.ylabel('ZCR')
plt.tight_layout()
plt.show()
```

Alternative plus simple avec numpy :

```python
def calculate_zcr(signal, frame_length=2048, hop_length=512):
    """Calcule le ZCR d'un signal audio"""
    # Découpage en trames
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    
    # Calcul du ZCR pour chaque trame
    signs = np.sign(frames)
    # Différence entre signes consécutifs
    diffs = np.abs(np.diff(signs, axis=0))
    # Comptage des changements de signe
    zcr = np.sum(diffs, axis=0) / (2 * frame_length)
    
    return zcr
```

## Limites et compléments

Bien que le ZCR soit un outil utile, il présente certaines limites :

- **Insensibilité à l'amplitude** : Le ZCR ne tient pas compte de l'amplitude du signal, seulement des changements de signe.
- **Sensibilité au bruit** : Un faible bruit peut augmenter significativement le ZCR, surtout dans les passages silencieux.
- **Simplicité** : Le ZCR seul n'est pas suffisant pour une analyse complète.

Pour une analyse plus robuste, le ZCR est souvent utilisé en complément d'autres caractéristiques :

- **RMS** (Root Mean Square) pour mesurer l'énergie du signal
- **MFCC** (Mel-Frequency Cepstral Coefficients) pour représenter le timbre
- **Spectrogramme** pour analyser le contenu fréquentiel au cours du temps
- **Centroïde spectral** pour caractériser la "brillance" d'un son

## Conclusion

Le Zero Crossing Rate est une mesure simple mais puissante pour caractériser les signaux audio. Sa facilité de calcul et son interprétation intuitive en font un outil de choix pour les premières étapes d'analyse audio, notamment pour distinguer différents types de sons ou détecter des transitions.

Si le ZCR ne peut à lui seul décrire toute la complexité d'un signal audio, il constitue une base solide dans la boîte à outils de traitement du signal, particulièrement pour les applications de classification et de segmentation audio.

Pour l'analyse audio avancée, il est généralement combiné avec d'autres descripteurs temporels et spectraux afin d'obtenir une caractérisation plus complète des sons. 
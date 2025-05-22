# Les coefficients cepstraux sur l'échelle de Mel (MFCC)

## Introduction

Les MFCC (Mel-Frequency Cepstral Coefficients) sont des descripteurs acoustiques largement utilisés pour représenter de manière compacte le spectre sonore. Ils constituent l'une des techniques les plus efficaces pour extraire des caractéristiques pertinentes à partir de signaux audio, en particulier pour les applications liées à la voix humaine.

Ces coefficients ont révolutionné plusieurs domaines comme :
- La reconnaissance vocale
- L'identification du locuteur
- La classification de genres musicaux
- La détection d'émotions dans la parole

L'intérêt majeur des MFCC réside dans leur capacité à modéliser la perception humaine des sons tout en réduisant considérablement la dimension des données à traiter.

## Principe de fonctionnement

Le calcul des MFCC suit un processus en plusieurs étapes qui transforme un signal audio brut en un ensemble compact de coefficients. Voici les principales étapes :

1. **Prétraitement du signal** : Normalisation et préaccentuation pour amplifier les hautes fréquences.

2. **Fenêtrage** : Division du signal en trames courtes (généralement 20-40ms) en appliquant une fenêtre (souvent de Hamming) pour éviter les discontinuités aux bords.

3. **Transformation de Fourier rapide (FFT)** : Conversion de chaque trame du domaine temporel au domaine fréquentiel, obtenant ainsi le spectre de puissance.

4. **Application de filtres sur l'échelle de Mel** : Passage du spectre à travers un banc de filtres triangulaires espacés selon l'échelle de Mel. Cette étape imite la perception non-linéaire des fréquences par l'oreille humaine.

5. **Application du logarithme** : Prise du logarithme des énergies obtenues à la sortie des filtres, ce qui reflète également la perception humaine qui est logarithmique en intensité.

6. **Transformation en cosinus discrète (DCT)** : Application d'une DCT pour décorréler les coefficients logarithmiques, produisant les MFCC finaux. Généralement, seuls les 13 à 20 premiers coefficients sont conservés.

![Schéma du processus MFCC](https://i.imgur.com/illustration_mfcc.png)

## Pourquoi utiliser l'échelle de Mel ?

L'échelle de Mel est une échelle perceptuelle qui modélise la façon dont l'oreille humaine perçoit les fréquences sonores. Cette échelle présente plusieurs avantages :

1. **Perception non-linéaire** : Notre oreille distingue mieux les changements de fréquences dans les basses fréquences que dans les hautes. L'échelle de Mel compresse les hautes fréquences et étire les basses, reflétant cette sensibilité différenciée.

2. **Formule de conversion** : La conversion d'une fréquence f (en Hz) vers l'échelle de Mel utilise généralement la formule :
   ```
   Mel(f) = 2595 * log10(1 + f/700)
   ```

3. **Pertinence biologique** : Cette transformation permet d'extraire des caractéristiques plus proches du traitement auditif humain, rendant les MFCC particulièrement efficaces pour les applications liées à la parole.

4. **Réduction de dimension** : En utilisant moins de filtres dans les hautes fréquences, on réduit la quantité d'informations tout en préservant les éléments perceptuellement significatifs.

## Utilisation des MFCC en pratique

Les MFCC sont devenus incontournables dans de nombreuses applications audio :

### Reconnaissance vocale (ASR)
Les systèmes de reconnaissance automatique de la parole utilisent les MFCC comme features d'entrée pour leurs modèles (HMM, réseaux de neurones, etc.). Ces coefficients capturent efficacement les caractéristiques phonétiques essentielles.

### Identification du locuteur
Les MFCC permettent de créer des "empreintes vocales" distinctives pour chaque personne, facilitant la reconnaissance de l'identité du locuteur.

### Classification musicale
Ils servent à extraire des caractéristiques pour classifier des genres musicaux, des instruments ou des émotions dans la musique.

### Détection d'anomalies acoustiques
Dans des contextes industriels ou médicaux, les MFCC peuvent aider à détecter des anomalies dans des machines ou des pathologies vocales.

### Systèmes de dialogue
Les assistants virtuels comme Siri, Alexa ou Google Assistant utilisent les MFCC dans leurs premières étapes de traitement vocal.

## Exemple de code

Voici un exemple simple en Python utilisant la bibliothèque librosa pour extraire les MFCC d'un fichier audio :

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Charger un fichier audio
fichier_audio = "exemple.wav"
signal, taux_echantillonnage = librosa.load(fichier_audio, sr=None)

# Extraire les MFCC
# n_mfcc: nombre de coefficients à extraire
# n_fft: taille de la fenêtre pour la FFT
# hop_length: décalage entre trames successives
mfccs = librosa.feature.mfcc(y=signal, 
                            sr=taux_echantillonnage, 
                            n_mfcc=13,
                            n_fft=2048, 
                            hop_length=512)

# Normalisation des MFCC (optionnel mais souvent utile)
mfccs_normalises = librosa.util.normalize(mfccs)

# Afficher les MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, 
                         x_axis='time', 
                         sr=taux_echantillonnage, 
                         hop_length=512)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()

# Utiliser les MFCC pour une tâche d'apprentissage automatique
# mfccs_moyennes = np.mean(mfccs, axis=1)  # Souvent utilisé comme feature simple
```

## Conclusion

Les MFCC constituent une technique puissante pour l'extraction de caractéristiques audio, particulièrement efficace pour les applications liées à la voix humaine. Leur force réside dans leur capacité à :
- Capturer l'information essentielle du signal audio
- Refléter les propriétés perceptuelles de l'audition humaine
- Réduire considérablement la dimension des données

Bien que les MFCC restent une référence dans le domaine, d'autres descripteurs sont parfois utilisés en complément ou comme alternatives :
- **Spectrogrammes** : représentation visuelle complète de l'évolution du spectre dans le temps
- **Coefficients Chroma** : utiles pour l'analyse musicale en regroupant les fréquences par classes de hauteur
- **LPCC** (Linear Prediction Cepstral Coefficients) : basés sur un modèle de prédiction linéaire
- **PLP** (Perceptual Linear Prediction) : intégrant davantage de propriétés perceptuelles

À l'ère de l'apprentissage profond, certains systèmes apprennent directement leurs représentations à partir des données brutes, mais les MFCC restent souvent utilisés comme features de référence ou comme point de départ, témoignant de leur robustesse et de leur efficacité. 
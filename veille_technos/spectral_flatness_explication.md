# Le Spectral Flatness en traitement du signal audio

## Introduction

Le Spectral Flatness (ou platitude spectrale en français) est un descripteur acoustique qui mesure le caractère tonal ou bruité d'un signal audio. Cette mesure quantifie à quel point la distribution d'énergie dans le spectre fréquentiel est uniforme ou concentrée. En d'autres termes, elle permet de déterminer si un son ressemble davantage à un bruit (spectre plat) ou à une note musicale (spectre avec des pics distincts).

Cette caractéristique est particulièrement utile pour analyser la texture sonore d'un signal et joue un rôle important dans de nombreuses applications de traitement audio, allant de la classification des sons à l'analyse musicale.

## Définition intuitive

Pour comprendre intuitivement le Spectral Flatness, imaginons deux types de sons extrêmes :

- **Un son tonal pur** (comme une note de flûte ou de diapason) : son énergie est concentrée autour de quelques fréquences spécifiques (la fondamentale et ses harmoniques). Son spectre présente des pics distincts et prononcés, avec peu d'énergie entre ces pics. Ce type de son a une faible platitude spectrale.

- **Un bruit blanc** : son énergie est répartie uniformément sur toutes les fréquences. Son spectre est "plat", sans pics ni creux significatifs. Ce type de son a une platitude spectrale élevée.

**Exemples concrets :**
- Une voyelle chantée ("aaa") aura une platitude spectrale faible
- Le son du vent ou de la pluie aura une platitude spectrale élevée
- Le son d'une cymbale se situe entre les deux, commençant avec une platitude moyenne-basse puis augmentant avec le temps

## Formule mathématique

Mathématiquement, le Spectral Flatness est défini comme le rapport entre la moyenne géométrique et la moyenne arithmétique du spectre de puissance :

```
SF = moyenne_géométrique(X[k]) / moyenne_arithmétique(X[k])
```

Plus précisément :

```
SF = exp(1/N * Σ log(X[k])) / (1/N * Σ X[k])
```

Où :
- X[k] représente les valeurs du spectre de puissance
- N est le nombre de bandes de fréquence
- Σ indique une somme sur toutes les bandes k

Cette formule peut sembler complexe, mais elle compare essentiellement deux façons de mesurer la tendance centrale du spectre :
- La moyenne géométrique est sensible aux variations relatives et pénalise fortement les creux dans le spectre
- La moyenne arithmétique est dominée par les valeurs les plus élevées

## Interprétation

Les valeurs du Spectral Flatness sont généralement exprimées sur une échelle de 0 à 1 (parfois convertie en décibels) :

- **Valeurs proches de 1** : indiquent un signal similaire à un bruit blanc, avec une énergie distribuée uniformément sur toutes les fréquences. Ces sons sont perçus comme bruités, sans hauteur tonale claire.

- **Valeurs proches de 0** : indiquent un signal très tonal, avec l'énergie concentrée sur quelques fréquences spécifiques. Ces sons sont perçus comme musicaux, avec une hauteur tonale définie.

**Analogies sonores :**
- Platitude proche de 1 : bruit de vagues, vent, friture électronique
- Platitude proche de 0 : note de piano, flûte, diapason
- Valeurs intermédiaires : voix parlée, instruments percussifs

En pratique, même les sons très tonals n'atteignent pas exactement 0, et même les bruits purs n'atteignent pas exactement 1.

## Applications courantes

Le Spectral Flatness est utilisé dans de nombreuses applications de traitement audio :

* **Classification audio** : distinction entre musique, parole, et bruits environnementaux
* **Détection de voix** : les segments de parole ont généralement une platitude plus basse que les bruits de fond
* **Analyse musicale** : identification des passages tonals vs. percussifs
* **Compression audio** : adaptation des algorithmes selon le type de contenu
* **Détection de transitions** : identification des changements entre sections musicales
* **Reconnaissance d'instruments** : certains instruments ont des signatures de platitude caractéristiques
* **Amélioration de la parole** : filtrage adaptatif basé sur le caractère tonal/bruité

## Exemple de code Python

Voici comment calculer le Spectral Flatness d'un signal audio en Python, en utilisant librosa :

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Charger un fichier audio
y, sr = librosa.load('audio_example.wav', sr=None)

# Calculer le Spectral Flatness
spectral_flatness = librosa.feature.spectral_flatness(y=y)

# Afficher la platitude spectrale moyenne
print(f"Spectral Flatness moyenne : {np.mean(spectral_flatness)}")

# Visualiser la platitude spectrale au cours du temps
plt.figure(figsize=(12, 4))
plt.plot(spectral_flatness[0])
plt.title('Spectral Flatness')
plt.xlabel('Trames')
plt.ylabel('Platitude')
plt.tight_layout()
plt.show()
```

Si vous souhaitez implémenter le calcul manuellement pour mieux comprendre :

```python
def calculate_spectral_flatness(y, n_fft=2048, hop_length=512):
    """Calcule la platitude spectrale manuellement"""
    # Calculer le spectrogramme (magnitude)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Calculer le spectre de puissance
    S_power = S**2
    
    # Pour chaque trame (colonne du spectrogramme)
    flatness = []
    for i in range(S_power.shape[1]):
        # Éviter les valeurs nulles avec un petit epsilon
        spec_frame = S_power[:, i] + 1e-10
        
        # Moyenne géométrique
        geo_mean = np.exp(np.mean(np.log(spec_frame)))
        
        # Moyenne arithmétique
        arith_mean = np.mean(spec_frame)
        
        # Spectral Flatness
        flatness.append(geo_mean / arith_mean)
    
    return np.array(flatness)
```

## Visualisation possible

La platitude spectrale est particulièrement utile lorsqu'elle est visualisée au cours du temps, car elle permet de repérer facilement les transitions entre passages tonals et bruiteux.

Une visualisation efficace consiste à superposer :
1. La forme d'onde du signal audio
2. Le spectrogramme (pour voir la distribution d'énergie)
3. La courbe de platitude spectrale

```python
# Code de visualisation
plt.figure(figsize=(12, 8))

# Forme d'onde
plt.subplot(3, 1, 1)
plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
plt.title('Forme d\'onde')
plt.ylabel('Amplitude')

# Spectrogramme
plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(
    librosa.stft(y), ref=np.max), 
    y_axis='log', x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme')

# Platitude spectrale
plt.subplot(3, 1, 3)
times = librosa.times_like(spectral_flatness, sr=sr)
plt.plot(times, spectral_flatness[0])
plt.title('Platitude spectrale')
plt.xlabel('Temps (s)')
plt.ylabel('Platitude')

plt.tight_layout()
plt.show()
```

Les segments avec une platitude élevée correspondent généralement à des passages bruités (consonnes fricatives dans la parole, cymbales dans la musique), tandis que les segments avec une platitude faible correspondent à des passages tonals (voyelles, notes tenues).

## Limites et compléments

Bien que le Spectral Flatness soit un outil puissant, il présente certaines limites :

- **Sensibilité au bruit de fond** : même un léger bruit peut augmenter significativement la platitude d'un signal tonal
- **Dépendance aux paramètres** : les résultats peuvent varier selon la taille de fenêtre FFT et le type de fenêtrage utilisé
- **Mesure unidimensionnelle** : réduit la complexité du spectre à une seule valeur
- **Insensibilité à la phase** : ne capture pas les informations de phase du signal

Pour une analyse plus complète, le Spectral Flatness est souvent utilisé en complément d'autres descripteurs :

- **Spectral Centroid** : centre de gravité du spectre (brillance)
- **Spectral Bandwidth** : étalement du spectre autour du centroïde
- **Spectral Contrast** : différence entre pics et vallées dans le spectre
- **MFCC** : coefficients cepstraux sur l'échelle de Mel (timbre)
- **Zero Crossing Rate** : taux de passage par zéro (complémentaire temporel)
- **RMS Energy** : énergie du signal

## Conclusion

Le Spectral Flatness est un descripteur acoustique essentiel qui permet de caractériser la texture spectrale d'un son sur l'axe tonal-bruité. Sa capacité à quantifier le degré d'organisation fréquentielle d'un signal en fait un outil précieux pour de nombreuses applications de traitement audio.

Sa simplicité conceptuelle (un spectre est-il plat ou pointu ?) combinée à sa puissance discriminative en fait un descripteur de choix pour les premières étapes d'analyse de signal, notamment pour segmenter des flux audio ou classifier des sons.

Dans les systèmes d'analyse audio modernes, la platitude spectrale est généralement intégrée dans un ensemble plus large de descripteurs pour capturer les multiples dimensions de la perception sonore. Néanmoins, elle reste l'un des meilleurs indicateurs pour répondre à la question fondamentale : "Ce son est-il musical ou bruité ?" 
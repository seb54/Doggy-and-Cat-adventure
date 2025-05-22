# Chroma STFT

## Introduction et contexte

L’analyse harmonique et mélodique est fondamentale en traitement du signal musical. L’un des outils principaux pour cette analyse est la **représentation chroma**, qui permet de visualiser la répartition de l’énergie sur les différentes hauteurs de notes. Le Chroma STFT, basé sur la transformée de Fourier à court terme, est très utilisé dans la recherche et les applications musicales.

---

## Résumé

Le Chroma STFT représente l’énergie d’un signal audio répartie sur les 12 notes de la gamme chromatique, indépendamment de l’octave. Cela permet d’identifier les notes et accords présents dans un extrait musical, facilitant l’analyse harmonique et la comparaison de morceaux.

---

## Utilité

* Analyse harmonique et détection des notes dominantes
* Reconnaissance d’accords, de tonalité, ou de mélodies
* Comparaison de morceaux ou alignement audio/partition

---

## Calcul

1. Calculer la STFT (transformée de Fourier à court terme).
2. Plier le spectre sur les 12 notes (chroma bins).
3. Additionner l’énergie pour chaque classe de note, toutes octaves confondues.

---

## Exemple de code (Python + librosa)

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Charger un fichier audio exemple
y, sr = librosa.load(librosa.ex('trumpet'))

# Calcul du chroma STFT
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Affichage du chromagramme
plt.figure(figsize=(12, 4))
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr, cmap='coolwarm')
plt.colorbar(label='Intensité')
plt.title('Chroma STFT')
plt.tight_layout()
plt.show()
```

---

## Pour aller plus loin

* [Documentation officielle Librosa](https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html)
* [Chroma feature Wikipedia (EN)](https://en.wikipedia.org/wiki/Chroma_feature)
* Recherches : ["chroma stft" Google Scholar](https://scholar.google.com/scholar?q=chroma+stft)

# Guide : Projet de Classification Audio Chien/Chat

## Introduction

Ce guide détaille l'implémentation d'un système modulaire de classification audio permettant de distinguer les sons de chiens et de chats. L'approche est progressive, partant d'une classification basée sur un seul critère jusqu'à l'intégration de multiples caractéristiques audio.

## Table des matières

1. [Préparation des données](#1-préparation-des-données)
2. [Transformation des données audio](#2-transformation-des-données-audio)
3. [Extraction des critères instantanés](#3-extraction-des-critères-instantanés)
4. [Agrégation des critères](#4-agrégation-des-critères)
5. [Classification avec un seul critère](#5-classification-avec-un-seul-critère)
6. [Évaluation comparative des critères](#6-évaluation-comparative-des-critères)
7. [Classification multi-critères progressive](#7-classification-multi-critères-progressive)
8. [Interface modulaire](#8-interface-modulaire)
9. [Optimisation et déploiement](#9-optimisation-et-déploiement)

## 1. Préparation des données

### Technologies
- **Python 3.x**
- **Librosa** : Manipulation audio
- **Système de fichiers** : Organisation des données

### Étapes
1. Collectez des échantillons audio de chiens et chats (minimum 200 par classe)
2. Organisez-les dans une structure de dossiers cohérente :
   ```
   /data
     /raw
       /chiens
       /chats
     /processed
   ```
3. Standardisez le format (WAV recommandé) et la fréquence d'échantillonnage (22050 Hz)

## 2. Transformation des données audio

### Technologies
- **Librosa** : Traitement audio
- **NumPy** : Manipulation de tableaux
- **SciPy** : Traitement du signal

### Code
```python
import librosa
import numpy as np
import os

def transformer_audio(fichier_audio, sr=22050, duree=2):
    """
    Charge et prépare un fichier audio pour l'extraction de caractéristiques
    
    Paramètres:
        fichier_audio: Chemin vers le fichier audio
        sr: Fréquence d'échantillonnage cible
        duree: Durée maximale à considérer (en secondes)
        
    Retourne:
        audio: Signal audio normalisé
        sr: Fréquence d'échantillonnage
        frame_length: Taille de trame pour l'analyse
        hop_length: Pas entre trames
    """
    # Charger le fichier audio
    audio, sr = librosa.load(fichier_audio, sr=sr, duration=duree)
    
    # Normaliser le signal
    audio = librosa.util.normalize(audio)
    
    # Paramètres pour l'analyse par trame
    frame_length = 1024  # ~46ms à 22050Hz
    hop_length = 512     # 50% de chevauchement
    
    return audio, sr, frame_length, hop_length

def traiter_ensemble_donnees(dossier_source, dossier_destination):
    """Prétraite tous les fichiers audio d'un dossier"""
    os.makedirs(dossier_destination, exist_ok=True)
    
    for fichier in os.listdir(dossier_source):
        if fichier.endswith(('.wav', '.mp3')):
            chemin_source = os.path.join(dossier_source, fichier)
            chemin_destination = os.path.join(dossier_destination, f"{os.path.splitext(fichier)[0]}.npy")
            
            # Transformer l'audio
            audio, sr, _, _ = transformer_audio(chemin_source)
            
            # Sauvegarder l'audio prétraité
            np.save(chemin_destination, audio)
            
    print(f"Prétraitement terminé. Fichiers sauvegardés dans {dossier_destination}")
```

## 3. Extraction des critères instantanés

### Technologies
- **Librosa** : Extraction de caractéristiques audio
- **NumPy** : Manipulation de données

### Code
```python
def extraire_criteres_instantanes(audio, sr, frame_length, hop_length):
    """
    Extrait les caractéristiques audio instantanées (par trame)
    
    Paramètres:
        audio: Signal audio
        sr: Fréquence d'échantillonnage
        frame_length: Taille de trame
        hop_length: Pas entre trames
        
    Retourne:
        Dictionnaire des caractéristiques instantanées
    """
    # Dictionnaire pour stocker les valeurs instantanées
    criteres_instantanes = {}
    
    # MFCC instantanés (par trame)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, 
                                 hop_length=hop_length, n_fft=frame_length)
    criteres_instantanes['mfcc'] = mfccs
    
    # RMS instantané (énergie)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
    criteres_instantanes['rms'] = rms
    
    # ZCR instantané (taux de passage par zéro)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)
    criteres_instantanes['zcr'] = zcr
    
    # Centroïde spectral instantané
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, 
                                                       n_fft=frame_length, hop_length=hop_length)
    criteres_instantanes['spectral_centroid'] = spectral_centroid
    
    # Largeur de bande spectrale instantanée
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, 
                                                n_fft=frame_length, hop_length=hop_length)
    criteres_instantanes['spectral_bandwidth'] = bandwidth
    
    # Platitude spectrale instantanée
    flatness = librosa.feature.spectral_flatness(y=audio, 
                                              n_fft=frame_length, hop_length=hop_length)
    criteres_instantanes['spectral_flatness'] = flatness
    
    # Rolloff spectral instantané
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, 
                                            n_fft=frame_length, hop_length=hop_length)
    criteres_instantanes['spectral_rolloff'] = rolloff
    
    # Chroma STFT instantané
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, 
                                       n_fft=frame_length, hop_length=hop_length)
    criteres_instantanes['chroma_stft'] = chroma
    
    # Tonnetz instantané (facultatif car nécessite harmonic/percussive)
    # harmonics = librosa.effects.harmonic(audio)
    # tonnetz = librosa.feature.tonnetz(y=harmonics, sr=sr)
    # criteres_instantanes['tonnetz'] = tonnetz
    
    return criteres_instantanes
```

## 4. Agrégation des critères

### Technologies
- **NumPy** : Calculs statistiques
- **Pandas** : Organisation des données

### Code
```python
import numpy as np
import pandas as pd

def agreger_criteres(criteres_instantanes):
    """
    Calcule des statistiques agrégées sur les caractéristiques instantanées
    
    Paramètres:
        criteres_instantanes: Dictionnaire des caractéristiques par trame
        
    Retourne:
        DataFrame des caractéristiques agrégées
    """
    # Dictionnaire pour stocker les valeurs agrégées
    criteres_agreges = {}
    
    # Pour chaque critère, calculer plusieurs statistiques
    for nom_critere, valeurs in criteres_instantanes.items():
        # Traitement spécial pour les MFCC (13 coefficients)
        if nom_critere == 'mfcc':
            for i in range(valeurs.shape[0]):  # Pour chaque coefficient
                criteres_agreges[f'mfcc_{i+1}_mean'] = np.mean(valeurs[i])
                criteres_agreges[f'mfcc_{i+1}_std'] = np.std(valeurs[i])
        # Pour les autres critères (unidimensionnels)
        else:
            criteres_agreges[f'{nom_critere}_mean'] = np.mean(valeurs)
            criteres_agreges[f'{nom_critere}_std'] = np.std(valeurs)
            criteres_agreges[f'{nom_critere}_min'] = np.min(valeurs)
            criteres_agreges[f'{nom_critere}_max'] = np.max(valeurs)
            criteres_agreges[f'{nom_critere}_median'] = np.median(valeurs)
    
    # Convertir en DataFrame pour faciliter la manipulation
    return pd.DataFrame([criteres_agreges])

def extraire_et_agreger_tous_fichiers(dossier_sources, etiquettes):
    """
    Extrait et agrège les caractéristiques de tous les fichiers
    
    Paramètres:
        dossier_sources: Liste des dossiers contenant les fichiers audio
        etiquettes: Étiquettes correspondantes (0 pour chien, 1 pour chat)
        
    Retourne:
        X: DataFrame des caractéristiques
        y: Étiquettes correspondantes
    """
    X_list = []
    y_list = []
    
    for dossier, etiquette in zip(dossier_sources, etiquettes):
        for fichier in os.listdir(dossier):
            if fichier.endswith(('.wav', '.mp3', '.npy')):
                chemin = os.path.join(dossier, fichier)
                
                # Charger et transformer l'audio
                if fichier.endswith('.npy'):
                    audio = np.load(chemin)
                    sr = 22050  # Fréquence supposée
                else:
                    audio, sr, _, _ = transformer_audio(chemin)
                
                frame_length = 1024
                hop_length = 512
                
                # Extraire les critères instantanés
                criteres = extraire_criteres_instantanes(audio, sr, frame_length, hop_length)
                
                # Agréger les critères
                criteres_agreges = agreger_criteres(criteres)
                
                # Ajouter aux listes
                X_list.append(criteres_agreges)
                y_list.append(etiquette)
    
    # Concaténer tous les DataFrames
    X = pd.concat(X_list, ignore_index=True)
    y = np.array(y_list)
    
    return X, y
```

## 5. Classification avec un seul critère

### Technologies
- **Scikit-learn** : Algorithmes de classification
- **Matplotlib/Seaborn** : Visualisation des résultats

### Code
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def classification_un_critere(X, y, nom_critere):
    """
    Classifie les sons en utilisant un seul critère
    
    Paramètres:
        X: DataFrame des caractéristiques
        y: Étiquettes (0 pour chien, 1 pour chat)
        nom_critere: Nom du critère à utiliser
        
    Retourne:
        Modèle entraîné et métriques de performance
    """
    # Sélectionner uniquement le critère spécifié
    X_single = X[[nom_critere]]
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.3, random_state=42)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraîner un modèle simple
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train_scaled, y_train)
    
    # Évaluation
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Chien', 'Chat'], 
                yticklabels=['Chien', 'Chat'])
    plt.title('Matrice de confusion')
    
    # Distribution des valeurs
    plt.subplot(1, 2, 2)
    for classe, nom in zip([0, 1], ['Chien', 'Chat']):
        sns.kdeplot(X_test_scaled[y_test == classe].flatten(), 
                   label=nom, shade=True)
    
    plt.title(f'Distribution de {nom_critere} par classe')
    plt.xlabel(nom_critere)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"Précision avec {nom_critere} uniquement: {accuracy:.2f}")
    print(report)
    
    return clf, accuracy, report
```

## 6. Évaluation comparative des critères

### Technologies
- **Scikit-learn** : Validation croisée
- **Pandas & Matplotlib** : Organisation et visualisation

### Code
```python
def evaluer_criteres_individuels(X, y, liste_criteres=None):
    """
    Évalue et compare la performance de chaque critère individuellement
    
    Paramètres:
        X: DataFrame des caractéristiques
        y: Étiquettes
        liste_criteres: Liste des critères à évaluer (si None, utilise toutes les colonnes de X)
        
    Retourne:
        DataFrame des performances triées
    """
    if liste_criteres is None:
        liste_criteres = X.columns.tolist()
    
    resultats = []
    
    for critere in liste_criteres:
        # Sélectionner le critère
        X_single = X[[critere]]
        
        # Normalisation et validation croisée
        scores = cross_val_score(
            Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', DecisionTreeClassifier(max_depth=3))
            ]), 
            X_single, y, cv=5, scoring='accuracy'
        )
        
        # Stocker les résultats
        resultats.append({
            'critere': critere,
            'precision_moyenne': scores.mean(),
            'ecart_type': scores.std()
        })
    
    # Convertir en DataFrame et trier
    df_resultats = pd.DataFrame(resultats).sort_values('precision_moyenne', ascending=False)
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    sns.barplot(x='critere', y='precision_moyenne', data=df_resultats.head(10),
               yerr=df_resultats.head(10)['ecart_type'])
    plt.title('Top 10 des critères les plus performants individuellement')
    plt.xlabel('Critère')
    plt.ylabel('Précision moyenne (validation croisée)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df_resultats
```

## 7. Classification multi-critères progressive

### Technologies
- **Scikit-learn** : Sélection de caractéristiques et classification
- **XGBoost/LightGBM** : Modèles avancés (optionnels)

### Code
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def classification_progressive(X, y, max_criteres=9):
    """
    Implémente une approche progressive d'ajout de critères
    
    Paramètres:
        X: DataFrame des caractéristiques
        y: Étiquettes
        max_criteres: Nombre maximum de critères à considérer
        
    Retourne:
        Résultats de performance par nombre de critères
    """
    # Évaluer l'importance des critères
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    
    # Récupérer les scores d'importance
    scores = pd.DataFrame({
        'critere': X.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    # Division train/test unique pour comparer équitablement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Résultats pour chaque nombre de critères
    resultats = []
    meilleurs_criteres = scores['critere'].tolist()
    
    for k in range(1, min(max_criteres + 1, len(meilleurs_criteres) + 1)):
        # Sélectionner les k meilleurs critères
        criteres_selectionnes = meilleurs_criteres[:k]
        X_train_k = X_train[criteres_selectionnes]
        X_test_k = X_test[criteres_selectionnes]
        
        # Pipeline avec normalisation et classificateur
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Entraînement
        pipeline.fit(X_train_k, y_train)
        
        # Évaluation
        y_pred = pipeline.predict(X_test_k)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Stocker les résultats
        resultats.append({
            'nb_criteres': k,
            'criteres': criteres_selectionnes,
            'precision': accuracy
        })
        
        print(f"Précision avec {k} critère(s): {accuracy:.4f}")
        print(f"Critères utilisés: {', '.join(criteres_selectionnes)}\n")
    
    # Visualisation de l'évolution de la précision
    df_resultats = pd.DataFrame(resultats)
    plt.figure(figsize=(10, 6))
    plt.plot(df_resultats['nb_criteres'], df_resultats['precision'], 'o-')
    plt.title('Évolution de la précision selon le nombre de critères')
    plt.xlabel('Nombre de critères')
    plt.ylabel('Précision')
    plt.grid(True)
    plt.xticks(df_resultats['nb_criteres'])
    plt.show()
    
    return df_resultats, pipeline
```

## 8. Interface modulaire

### Technologies
- **Flask/FastAPI** : API REST
- **Joblib** : Sauvegarde/chargement de modèles
- **JSON** : Format d'échange

### Code
```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import io
import numpy as np
import tempfile
import os

# Charger les modèles pré-entraînés
modele_un_critere = joblib.load('modeles/modele_un_critere.joblib')
modele_multi_criteres = joblib.load('modeles/modele_multi_criteres.joblib')

app = FastAPI(title="API Classification Audio Chien/Chat")

@app.post("/predict/single-feature/")
async def predict_single_feature(file: UploadFile = File(...)):
    """Prédiction avec un seul critère"""
    # Sauvegarder temporairement le fichier audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    
    try:
        # Transformer l'audio
        audio, sr, frame_length, hop_length = transformer_audio(temp_path)
        
        # Extraire les critères instantanés
        criteres = extraire_criteres_instantanes(audio, sr, frame_length, hop_length)
        
        # Agréger les critères
        criteres_agreges = agreger_criteres(criteres)
        
        # Sélectionner le critère utilisé pour ce modèle
        X = criteres_agreges[['zcr_mean']]  # Exemple avec ZCR
        
        # Prédire
        prediction = modele_un_critere.predict(X)[0]
        proba = modele_un_critere.predict_proba(X)[0]
        
        # Déterminer la classe
        classe = 'chat' if prediction == 1 else 'chien'
        
        return {
            "classe_predite": classe,
            "probabilite": float(max(proba)),
            "critere_utilise": "zcr_mean"
        }
    finally:
        # Nettoyer le fichier temporaire
        os.unlink(temp_path)

@app.post("/predict/multi-feature/")
async def predict_multi_feature(
    file: UploadFile = File(...),
    nb_criteres: int = Form(5)
):
    """Prédiction avec plusieurs critères"""
    # Sauvegarder temporairement le fichier audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name
    
    try:
        # Transformer l'audio
        audio, sr, frame_length, hop_length = transformer_audio(temp_path)
        
        # Extraire les critères instantanés
        criteres = extraire_criteres_instantanes(audio, sr, frame_length, hop_length)
        
        # Agréger les critères
        criteres_agreges = agreger_criteres(criteres)
        
        # Sélectionner les critères selon le nombre demandé
        meilleurs_criteres = [
            'zcr_mean', 'spectral_centroid_mean', 'mfcc_1_mean',
            'spectral_bandwidth_mean', 'rms_mean', 'mfcc_2_mean',
            'spectral_rolloff_mean', 'spectral_flatness_mean', 'chroma_stft_mean'
        ][:nb_criteres]
        
        X = criteres_agreges[meilleurs_criteres]
        
        # Prédire
        prediction = modele_multi_criteres.predict(X)[0]
        proba = modele_multi_criteres.predict_proba(X)[0]
        
        # Déterminer la classe
        classe = 'chat' if prediction == 1 else 'chien'
        
        return {
            "classe_predite": classe,
            "probabilite": float(max(proba)),
            "nb_criteres_utilises": nb_criteres,
            "criteres_utilises": meilleurs_criteres
        }
    finally:
        # Nettoyer le fichier temporaire
        os.unlink(temp_path)

# Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 9. Optimisation et déploiement

### Technologies
- **Docker** : Conteneurisation
- **MLflow** : Suivi d'expériences (optionnel)
- **TensorFlow Lite/ONNX** : Optimisation pour le déploiement

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système pour librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier les modèles et le code
COPY modeles/ ./modeles/
COPY *.py .

# Exposer le port utilisé par l'API
EXPOSE 8000

# Lancer l'API
CMD ["python", "api.py"]
```

### requirements.txt
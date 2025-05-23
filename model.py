import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Chargement des données
def load_data():
    train_data = pd.read_csv('data/moyennes_train.csv')
    test_data = pd.read_csv('data/moyennes_test.csv')
    
    # Séparation des features et des labels
    X_train = train_data.drop('classe', axis=1).values
    X_test = test_data.drop('classe', axis=1).values
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Encodage des labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_data['classe'])
    y_test = le.transform(test_data['classe'])
    
    # Conversion en format catégorique
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Chargement des données
    X_train, X_test, y_train, y_test = load_data()
    
    # Création du modèle
    input_shape = (X_train.shape[1],)
    num_classes = y_train.shape[1]
    model = create_model(input_shape, num_classes)
    
    # Configuration de l'early stopping
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    # Entraînement du modèle
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Évaluation du modèle
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nPrécision sur l'ensemble de test : {test_accuracy:.4f}")

if __name__ == '__main__':
    main() 

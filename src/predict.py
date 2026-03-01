import pandas as pd
import numpy as np
import joblib
import os
import sys

# Configuration du chemin pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_prediction_tools():
    """Charge le modèle, le scaler et la PCA sauvegardés"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        pca = joblib.load('models/pca_model.pkl')
        return model, scaler, pca
    except FileNotFoundError as e:
        print(f"❌ Erreur : Fichiers modèles introuvables ({e})")
        return None, None, None

def predict_churn(new_data):
    """
    Prend un DataFrame de nouvelles données, applique les transformations
    et retourne les prédictions.
    """
    model, scaler, pca = load_prediction_tools()
    if model is None: return
    
    # 1. Mise à l'échelle (Scaling)
    # Important : Utiliser le même scaler que l'entraînement
    data_scaled = scaler.transform(new_data)
    
    # 2. Transformation PCA
    # On réduit les colonnes aux 10 composantes PC1...PC10
    data_pca = pca.transform(data_scaled)
    
    # 3. Prédiction
    predictions = model.predict(data_pca)
    probabilities = model.predict_proba(data_pca)[:, 1]
    
    return predictions, probabilities

def main():
    print("🔮 Module de Prédiction de Churn")
    print("-" * 30)

    # Exemple : Chargement d'un petit échantillon pour tester
    # Normalement, ici tu chargerais tes nouvelles données clients
    try:
        X_test = pd.read_csv('data/train_test/X_test.csv') # Juste pour le test
        # Note : X_test.csv est déjà en PCA, donc pour un vrai test 'unitaire', 
        # il faudrait des données brutes (raw). 
        # Mais voici comment afficher les résultats :
        
        model = joblib.load('models/best_model.pkl')
        
        # Simulation d'une prédiction sur les 5 premiers clients du test set
        sample = X_test.head(10)
        preds = model.predict(sample)
        probs = model.predict_proba(sample)[:, 1]

        results = pd.DataFrame({
            'Client_ID': range(1, 11),
            'Prediction': ['CHURN' if p == 1 else 'FIDELE' for p in preds],
            'Probabilité_Churn': [f"{p*100:.2f}%" for p in probs]
        })

        print("\n🚀 Résultats des prédictions :")
        print(results.to_string(index=False))

    except Exception as e:
        print(f"⚠️ Erreur lors du test : {e}")

if __name__ == "__main__":
    main()
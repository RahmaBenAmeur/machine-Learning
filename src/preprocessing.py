import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def drop_high_cardinality(df, threshold=0.90):
    """Supprime les colonnes avec trop de valeurs uniques (ID-like)"""
    cols_to_drop = []
    for col in df.columns:
        if col not in ['Churn', 'MonetaryTotal']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > threshold:
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)

def clean_and_prepare_data(file_path):
    if not os.path.exists(file_path):
        print(f" Erreur : Le fichier {file_path} est introuvable.")
        return

    df = pd.read_csv(file_path)
    print(f" Données initiales : {len(df)} lignes")

    # --- 1. OUTLIERS & DATES ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_for_outlier = [c for c in numeric_cols if c not in ['Churn', 'CustomerID']]
    iso_forest = IsolationForest(contamination=0.06, random_state=42)
    outlier_preds = iso_forest.fit_predict(df[features_for_outlier].fillna(0))
    df = df[outlier_preds == 1].reset_index(drop=True)

    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear'] = df['RegistrationDate'].dt.year.fillna(df['RegistrationDate'].dt.year.median())
    df['RegMonth'] = df['RegistrationDate'].dt.month.fillna(df['RegistrationDate'].dt.month.median())

    # --- 2. GESTION DES CIBLES & NETTOYAGE ---
    y_reg_full = df['MonetaryTotal'].fillna(df['MonetaryTotal'].median())
    
    cols_to_drop = [
        'Recency', 'AccountStatus', 'RFMSegment', 'ChurnRiskCategory', 
        'CustomerID', 'RegistrationDate', 'LastLoginIP', 'NewsletterSubscribed',
        'MonetaryAvg', 'TotalQuantity', 'TotalTransactions'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = drop_high_cardinality(df)

    # Encodage des variables catégorielles
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.median())

    # --- 3. PCA (Réduction de dimension) ---
    X_raw = df.drop(columns=['Churn', 'MonetaryTotal'], errors='ignore')
    y_class = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création du DataFrame final avec les 10 composantes
    X_final = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])

    # --- 4. CLUSTERING SUR PCA (C'est ici que l'erreur se corrige !) ---
    print(" Entraînement du KMeans sur les 10 composantes PCA...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_final) # On entraîne sur 10 features, donc il attendra 10 features au predict

    # --- 5. SAUVEGARDE DES DONNÉES PROCESSED ---
    os.makedirs('data/processed', exist_ok=True)
    df_processed_all = X_final.copy()
    df_processed_all['Churn'] = y_class.values
    df_processed_all['MonetaryTotal'] = y_reg_full.values
    df_processed_all.to_csv('data/processed/processed_data_final.csv', index=False)

    # --- 6. SPLIT & EXPORT ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    y_train_reg, y_test_reg = train_test_split(
        y_reg_full, test_size=0.2, random_state=42
    )

    os.makedirs('data/train_test', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)
    y_train_reg.to_csv('data/train_test/y_reg_train.csv', index=False)
    y_test_reg.to_csv('data/train_test/y_reg_test.csv', index=False)
    
    # Sauvegarde des objets (modèles de préparation)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca_model.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl') # Ce fichier est maintenant correct !
    
    print(f" Preprocessing terminé. PCA (10 colonnes) et KMeans synchronisés.")

if __name__ == "__main__":
    clean_and_prepare_data('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
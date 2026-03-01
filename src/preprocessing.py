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
        if col != 'Churn':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > threshold:
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)

def clean_and_prepare_data(file_path):
    # 1. Chargement
    df = pd.read_csv(file_path)
    print(f"📊 Données initiales : {len(df)} lignes")

    # ==========================================
    # ÉTAPE 1 : OUTLIERS & DATES
    # ==========================================
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_for_outlier = [c for c in numeric_cols if c not in ['Churn', 'CustomerID']]
    
    iso_forest = IsolationForest(contamination=0.06, random_state=42)
    outlier_preds = iso_forest.fit_predict(df[features_for_outlier].fillna(0))
    df = df[outlier_preds == 1].reset_index(drop=True)

    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear'] = df['RegistrationDate'].dt.year.fillna(df['RegistrationDate'].dt.year.median())
    df['RegMonth'] = df['RegistrationDate'].dt.month.fillna(df['RegistrationDate'].dt.month.median())

    # ==========================================
    # ÉTAPE 2 : NETTOYAGE (Anti-Leakage & Redondance)
    # ==========================================
    cols_to_drop = [
        'Recency', 'AccountStatus', 'RFMSegment', 'ChurnRiskCategory', 'TenureRatio',
        'CustomerID', 'RegistrationDate', 'LastLoginIP', 'NewsletterSubscribed',
        'MonetaryTotal', 'MonetaryAvg', 'TotalQuantity', 'TotalTransactions'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = drop_high_cardinality(df)

    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    df = df.fillna(df.median())

    # ==========================================
    # ÉTAPE 3 : CLUSTERING (K-MEANS)
    # ==========================================
    print("🤖 Application du Clustering (K-Means)...")
    X_temp = df.drop(columns=['Churn'])
    scaler_temp = StandardScaler()
    X_temp_scaled = scaler_temp.fit_transform(X_temp)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_Segment'] = kmeans.fit_predict(X_temp_scaled)

    # ==========================================
    # ÉTAPE 4 : PCA (RÉDUCTION DE DIMENSION)
    # ==========================================
    print("📉 Réduction via PCA (Objectif : 10 composantes)...")
    X_raw = df.drop(columns=['Churn'])
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création du DataFrame final pré-traité
    X_final = pd.DataFrame(
        X_pca, 
        columns=[f'PC{i+1}' for i in range(10)]
    )
    
    # Dataset complet (X + y) pour sauvegarde
    processed_full_df = X_final.copy()
    processed_full_df['Churn'] = y.values

    # ==========================================
    # ÉTAPE 5 : SAUVEGARDE DU DATASET COMPLET (AVANT SPLIT)
    # ==========================================
    os.makedirs('data/processed', exist_ok=True)
    processed_full_df.to_csv('data/processed/processed_data.csv', index=False)
    print(f"💾 Dataset complet sauvegardé dans : data/processed/processed_data.csv")

    # ==========================================
    # ÉTAPE 6 : SPLIT TRAIN/TEST & SAUVEGARDE
    # ==========================================
    print("✂️ Découpage Train/Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs('data/train_test', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)
    
    # Sauvegarde des modèles de transformation
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca_model.pkl')
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    
    print(f"✅ Opération terminée ! {X_train.shape[1]} composantes PCA générées.")

if __name__ == "__main__":
    clean_and_prepare_data('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

def clean_and_prepare_data(file_path):
    # 1. Chargement
    df = pd.read_csv(file_path)
    initial_rows = df.shape[0]
    print(f"Données initiales : {initial_rows} lignes")

    # ==========================================
    # ÉTAPE 1 : SUPPRESSION OUTLIERS (ISOLATION FOREST)
    # ==========================================
    print("\n Suppression des outliers (contamination=0.06)...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_for_outlier = [c for c in numeric_cols if c not in ['Churn', 'CustomerID']]
    
    # Remplissage temporaire des NaN pour Isolation Forest
    df_for_iso = df[features_for_outlier].fillna(df[features_for_outlier].median())
    
    iso_forest = IsolationForest(contamination=0.06, random_state=42, n_estimators=100)
    outlier_preds = iso_forest.fit_predict(df_for_iso)
    
    # Garder uniquement les inliers
    df = df[outlier_preds == 1].reset_index(drop=True)
    n_outliers = initial_rows - len(df)
    
    print(f"   → {n_outliers} outliers supprimés ({n_outliers/initial_rows*100:.1f}%)")
    print(f"   → {len(df)} lignes conservées")

    # ==========================================
    # ÉTAPE 2 : PARSING DATES
    # ==========================================
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    df['RegYear'] = df['RegistrationDate'].dt.year
    df['RegMonth'] = df['RegistrationDate'].dt.month
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday

    # ==========================================
    # ÉTAPE 3 : IMPUTATION
    # ==========================================
    print("\n Imputation...")
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(df['AvgDaysBetweenPurchases'].median())
    df['SupportTicketsCount'] = df['SupportTicketsCount'].fillna(df['SupportTicketsCount'].mean())
    df['SatisfactionScore'] = df['SatisfactionScore'].fillna(df['SatisfactionScore'].mean())
    
    for col in ['Gender', 'AccountStatus']:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # ==========================================
    # ÉTAPE 4 : FEATURE ENGINEERING
    # ==========================================
    print("Feature Engineering...")
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'].replace(0, 1))
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    # ==========================================
    # ÉTAPE 5 : SUPPRESSION COLONNES INUTILES
    # ==========================================
    print("Suppression colonnes inutiles...")
    # NewsletterSubscribed = 100%  (constante)
    # RegistrationDate = remplacée par features temporelles
    # CustomerID 
    # LastLoginIP = donnée brute complexe
    cols_to_drop = ['NewsletterSubscribed', 'RegistrationDate', 'CustomerID', 'LastLoginIP']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ==========================================
    # ÉTAPE 6 : ENCODAGE
    # ==========================================
    print(" Encodage...")
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # ==========================================
    # ÉTAPE 7 : STANDARDISATION
    # ==========================================
    print(" Standardisation...")
    scaler = StandardScaler()
    features_to_scale = [c for c in df.columns if c != 'Churn']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Sauvegarde
    df.to_csv('data/processed/processed_data.csv', index=False)
    print(f"\n Terminé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    return df

def split_and_save(df):
    print("\n Split Train/Test (80/20)...")
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    
    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)
    print(" Sauvegardé!")

if __name__ == "__main__":
    import os
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/train_test', exist_ok=True)
    
    data_cleaned = clean_and_prepare_data('data/raw/retail_customers_COMPLETE_CATEGORICAL.csv')
    split_and_save(data_cleaned)
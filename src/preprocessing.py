import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def clean_and_prepare_data(file_path):
    # 1. Chargement des données
    df = pd.read_csv(file_path)
    
    
    # 2. Nettoyage des données
    df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
    
    # Extraction de nouvelles variables temporelles
    df['RegYear'] = df['RegistrationDate'].dt.year
    df['RegMonth'] = df['RegistrationDate'].dt.month
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday

    # Médiane pour données asymétriques ( Age)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(df['AvgDaysBetweenPurchases'].median())

    # Moyenne pour données numériques (Support Tickets)
    df['SupportTicketsCount'] = df['SupportTicketsCount'].fillna(df['SupportTicketsCount'].mean())

    # Mode pour données catégorielles
    for col in ['Gender', 'AccountStatus']:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Feature Engineering (Création de ratios métier)
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'].replace(0, 1))


    # 5. Suppression des colonnes inutiles ou constantes ('NewsletterSubscribed')
    cols_to_drop = ['NewsletterSubscribed', 'RegistrationDate', 'CustomerID', 'LastLoginIP']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 6. Encodage des catégories (Texte -> Chiffres)
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 7. Standardisation (StandardScaler) - CRUCIAL pour le K-means
    # On normalise tout sauf la cible 'Churn'
    scaler = StandardScaler()
    features_to_scale = df.columns.drop('Churn') if 'Churn' in df.columns else df.columns
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    df.to_csv('data/processed/processed_data.csv', index=False)
    print("Données nettoyées sauvegardées dans data/processed/processed_data.csv")
    
    return df

def split_and_save(df):
    # 8. Split Train/Test (80/20 avec stratification sur Churn)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Sauvegarde des fichiers pour les étapes suivantes
    X_train.to_csv('data/train_test/X_train.csv', index=False)
    X_test.to_csv('data/train_test/X_test.csv', index=False)
    y_train.to_csv('data/train_test/y_train.csv', index=False)
    y_test.to_csv('data/train_test/y_test.csv', index=False)
    
    print("Prétraitement terminé. Fichiers sauvegardés dans data/train_test/")

# Utilisation
if __name__ == "__main__":
    input_file = 'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv'
    data_cleaned = clean_and_prepare_data(input_file)
    split_and_save(data_cleaned)
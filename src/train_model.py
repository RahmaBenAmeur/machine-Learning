import pandas as pd
import numpy as np
import joblib
import os
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Import de tes fonctions personnalisées
from utils import (
    save_model, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, calculate_metrics,
    get_resampling_strategy, compare_models
)

warnings.filterwarnings('ignore')

DATA_PATHS = {
    'X_train': 'data/train_test/X_train.csv',
    'X_test': 'data/train_test/X_test.csv',
    'y_train': 'data/train_test/y_train.csv',
    'y_test': 'data/train_test/y_test.csv',
    'y_train_reg': 'data/train_test/y_reg_train.csv',
    'y_test_reg': 'data/train_test/y_reg_test.csv'
}

def main():
    print(" Démarrage de l'entraînement (Classification & Régression)...")
    
    # 1. Chargement
    try:
        X_train = pd.read_csv(DATA_PATHS['X_train'])
        X_test = pd.read_csv(DATA_PATHS['X_test'])
        y_train = pd.read_csv(DATA_PATHS['y_train']).values.ravel()
        y_test = pd.read_csv(DATA_PATHS['y_test']).values.ravel()
        y_train_reg = pd.read_csv(DATA_PATHS['y_train_reg']).values.ravel()
        y_test_reg = pd.read_csv(DATA_PATHS['y_test_reg']).values.ravel()
    except FileNotFoundError as e:
        print(f" Erreur: {e}. Lancez preprocessing.py d'abord.")
        return

    # 2. SMOTE (Gestion du déséquilibre)
    print("\n Application de SMOTE sur les données d'entraînement...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 3. Comparaison Classifieurs
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(criterion="entropy", random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    os.makedirs('reports', exist_ok=True)
    
    for name, clf in classifiers.items():
        print(f"--- Entraînement de {name} ---")
        clf.fit(X_res, y_res)
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        results[name] = metrics
        
        plot_confusion_matrix(y_test, y_pred, save_path=f'reports/cm_{name}.png')
        plot_roc_curve(y_test, y_proba, save_path=f'reports/roc_{name}.png')

    # 4. Meilleur modèle Classification
    compare_models(results)
    best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
    print(f"\n Meilleur modèle retenu : {best_model_name}")
    save_model(classifiers[best_model_name], 'models/best_model.pkl')

    # 5. Régression
    print("\n--- Entraînement de la Régression (Spending) ---")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    save_model(reg_model, 'models/regression_model.pkl')
    
    print("\n Tous les modèles sont prêts dans /models et les rapports dans /reports.")

if __name__ == "__main__":
    main()
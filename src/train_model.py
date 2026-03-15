import pandas as pd
import numpy as np
import sys
import os
import shutil
import joblib
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# Import de tes fonctions personnalisées
from utils import (
    save_model, save_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, print_classification_report, calculate_metrics,
    get_resampling_strategy, compare_models
)

warnings.filterwarnings('ignore')

# Configuration des chemins
DATA_PATHS = {
    'X_train': 'data/train_test/X_train.csv',
    'X_test': 'data/train_test/X_test.csv',
    'y_train': 'data/train_test/y_train.csv',
    'y_test': 'data/train_test/y_test.csv',
    'y_train_reg': 'data/train_test/y_reg_train.csv',
    'y_test_reg': 'data/train_test/y_reg_test.csv'
}

def main():
    print("🚀 Démarrage de l'entraînement (Méthode ENIS)...")
    
    # 1. Chargement des données
    try:
        X_train = pd.read_csv(DATA_PATHS['X_train'])
        X_test = pd.read_csv(DATA_PATHS['X_test'])
        y_train = pd.read_csv(DATA_PATHS['y_train']).values.ravel()
        y_test = pd.read_csv(DATA_PATHS['y_test']).values.ravel()
        y_train_reg = pd.read_csv(DATA_PATHS['y_train_reg']).values.ravel()
        y_test_reg = pd.read_csv(DATA_PATHS['y_test_reg']).values.ravel()
    except FileNotFoundError:
        print("❌ Erreur: Fichiers de données introuvables. Lancez preprocessing.py d'abord.")
        return

    # 2. Analyse et Gestion du déséquilibre (SMOTE - comme vu en classe)
    get_resampling_strategy(y_train)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 3. COMPARAISON DES CLASSIFIEURS (Comme l'exercice Sonar)
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(criterion="entropy", random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n--- Entraînement de {name} ---")
        clf.fit(X_res, y_res) # Apprentissage sur données équilibrées
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Utilisation de tes fonctions utils.py
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        results[name] = metrics
        
        plot_confusion_matrix(y_test, y_pred, save_path=f'reports/cm_{name}.png')
        plot_roc_curve(y_test, y_proba, save_path=f'reports/roc_{name}.png')
        
        if hasattr(clf, 'feature_importances_'):
            plot_feature_importance(clf, X_train.columns, save_path=f'reports/feat_{name}.png')

    # 4. Sélection et sauvegarde du meilleur modèle (basé sur F1-Score)
    compare_models(results)
    best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
    print(f"\n🏆 Meilleur modèle retenu : {best_model_name}")
    save_model(classifiers[best_model_name], 'models/best_model.pkl')

    # 5. RÉGRESSION (Prédire les dépenses futures)
    print("\n--- Entraînement de la Régression (Spending Prediction) ---")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_reg)
    save_model(reg_model, 'models/regression_model.pkl')
    
    # 6. CLUSTERING (Justification de l'Inertia - Exercice de classe)
    # On rappelle que le modèle KMeans a été créé au preprocessing
    print("\n✅ Modèles de Classification, Régression et Clustering prêts dans /models.")

if __name__ == "__main__":
    main()
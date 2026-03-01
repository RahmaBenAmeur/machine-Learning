import pandas as pd
import numpy as np
import sys
import os
import shutil
import joblib
import warnings
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Configuration du chemin pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    save_model, save_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, print_classification_report, calculate_metrics,
    get_resampling_strategy, compare_models
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATHS = {
    'X_train': 'data/train_test/X_train.csv',
    'X_test': 'data/train_test/X_test.csv',
    'y_train': 'data/train_test/y_train.csv',
    'y_test': 'data/train_test/y_test.csv'
}

def create_resampler():
    return SMOTE(random_state=42, k_neighbors=5)

# ============================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("RANDOM FOREST + PCA COMPONENTS")
    print("="*60)
    
    pipeline = ImbPipeline([
        ('smote', create_resampler()),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [5, 8, 12], 
        'rf__min_samples_leaf': [2, 4],
        'rf__class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # --- ADDED VISUAL METRICS ---
    plot_confusion_matrix(y_test, y_pred, save_path='reports/confusion_matrix_rf.png')
    plot_roc_curve(y_test, y_pred_proba, save_path='reports/roc_curve_rf.png')
    # ----------------------------
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_classification_report(y_test, y_pred)
    
    plot_feature_importance(best_model.named_steps['rf'], X_train.columns, 
                           top_n=len(X_train.columns),
                           save_path='reports/feature_importance_rf.png')
    
    return best_model, metrics, grid_search.best_params_

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("GRADIENT BOOSTING + PCA COMPONENTS")
    print("="*60)
    
    pipeline = ImbPipeline([
        ('smote', create_resampler()),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])
    
    param_grid = {
        'gb__n_estimators': [100, 150],
        'gb__learning_rate': [0.05, 0.1],
        'gb__max_depth': [3, 4]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # --- ADDED VISUAL METRICS ---
    plot_confusion_matrix(y_test, y_pred, save_path='reports/confusion_matrix_gb.png')
    plot_roc_curve(y_test, y_pred_proba, save_path='reports/roc_curve_gb.png')
    # ----------------------------
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_classification_report(y_test, y_pred)
    
    plot_feature_importance(best_model.named_steps['gb'], X_train.columns, 
                           top_n=len(X_train.columns),
                           save_path='reports/feature_importance_gb.png')
    
    return best_model, metrics, grid_search.best_params_

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION + PCA COMPONENTS")
    print("="*60)
    
    pipeline = ImbPipeline([
        ('smote', create_resampler()),
        ('lr', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
    ])
    
    param_grid = {'lr__C': [0.1, 1, 10], 'lr__penalty': ['l1', 'l2']}
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # --- ADDED VISUAL METRICS ---
    plot_confusion_matrix(y_test, y_pred, save_path='reports/confusion_matrix_lr.png')
    plot_roc_curve(y_test, y_pred_proba, save_path='reports/roc_curve_lr.png')
    # ----------------------------
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_classification_report(y_test, y_pred)
    
    return best_model, metrics, grid_search.best_params_

def main():
    print("🚀 Démarrage de l'entraînement sur données PCA...")
    try:
        X_train = pd.read_csv(DATA_PATHS['X_train'])
        X_test = pd.read_csv(DATA_PATHS['X_test'])
        y_train = pd.read_csv(DATA_PATHS['y_train']).values.ravel()
        y_test = pd.read_csv(DATA_PATHS['y_test']).values.ravel()
    except FileNotFoundError:
        print("❌ Erreur : Fichiers PCA introuvables.")
        return

    results = {}
    rf_model, rf_metrics, _ = train_random_forest(X_train, y_train, X_test, y_test)
    results['Random Forest'] = rf_metrics
    save_model(rf_model, 'models/random_forest.pkl')

    gb_model, gb_metrics, _ = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results['Gradient Boosting'] = gb_metrics
    save_model(gb_model, 'models/gradient_boosting.pkl')

    lr_model, lr_metrics, _ = train_logistic_regression(X_train, y_train, X_test, y_test)
    results['Logistic Regression'] = lr_metrics
    save_model(lr_model, 'models/logistic_regression.pkl')

    print("\n" + "="*60)
    print("RÉSULTATS FINAUX")
    print("="*60)
    compare_models(results)
    
    best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
    shutil.copy(f'models/{best_model_name.lower().replace(" ", "_")}.pkl', 'models/best_model.pkl')
    print(f"\n🏆 Meilleur modèle : {best_model_name}")

if __name__ == "__main__":
    main()
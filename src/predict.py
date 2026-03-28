import pandas as pd
import joblib
import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

# Configuration des chemins vers les modèles sauvegardés
MODEL_PATHS = {
    'classifier': 'models/best_model.pkl',
    'regressor': 'models/regression_model.pkl',
    'kmeans': 'models/kmeans_model.pkl'
}


DATA_TEST_PATH = 'data/train_test/X_test.csv'
TARGET_TEST_PATH = 'data/train_test/y_test.csv'

def run_comprehensive_predictions():
    print(" Lancement des tests multi-modèles (Classification, Régression, Clustering)...")

    # 1. Chargement des modèles
    try:
        classifier = joblib.load(MODEL_PATHS['classifier'])
        regressor = joblib.load(MODEL_PATHS['regressor'])
        kmeans = joblib.load(MODEL_PATHS['kmeans'])
        print(" Tous les modèles ont été chargés avec succès.")
    except FileNotFoundError as e:
        print(f" Erreur : Fichier modèle introuvable. {e}")
        print("Assurez-vous d'avoir lancé preprocessing.py puis train_model.py.")
        return

    # 2. Chargement des données de test
    if not os.path.exists(DATA_TEST_PATH):
        print(f" Erreur : {DATA_TEST_PATH} introuvable.")
        return
        
    X_test = pd.read_csv(DATA_TEST_PATH)
    y_true = pd.read_csv(TARGET_TEST_PATH).values.ravel()

    print(f" Test sur {len(X_test)} clients avec {X_test.shape[1]} features PCA.")

    
    print(" Calcul des prédictions en cours...")
    
    # --- Classification (Churn) ---
    churn_preds = classifier.predict(X_test)
    churn_probs = classifier.predict_proba(X_test)[:, 1]

    # --- Régression (Dépenses) ---
    spending_preds = regressor.predict(X_test)

    # --- Clustering (Segments) ---
    # On utilise .values pour éviter le warning sur les noms de colonnes
    customer_clusters = kmeans.predict(X_test.values)

    # 4. Compilation des résultats
    results = pd.DataFrame({
        'Real_Status': y_true,
        'Predicted_Churn': churn_preds,
        'Churn_Probability_%': (churn_probs * 100).round(2),
        'Predicted_Spending_DT': spending_preds.round(2),
        'Customer_Segment': customer_clusters
    })

    # Mappage des segments pour la présentation
    segment_map = {0: "Econome", 1: "VIP", 2: "Occasionnel", 3: "A risque"}
    results['Segment_Name'] = results['Customer_Segment'].map(segment_map)

    # 5. Sauvegarde des résultats
    os.makedirs('data/results', exist_ok=True)
    output_path = 'data/results/test_predictions_complet.csv'
    results.to_csv(output_path, index=False)

    # 6. Affichage du résumé
    print("\n" + "="*50)
    print(" RÉSUMÉ DES TESTS")
    print("="*50)
    print(f"Nombre de clients analysés : {len(results)}")
    print(f"Clients détectés 'Churn'  : {results['Predicted_Churn'].sum()}")
    print(f"Dépense moyenne prévue    : {results['Predicted_Spending_DT'].mean():.2f} DT")
    print("\nRépartition par Segment :")
    print(results['Segment_Name'].value_counts())
    print("="*50)
    
    print(f"\n Résultats détaillés : {output_path}")

if __name__ == "__main__":
    run_comprehensive_predictions()
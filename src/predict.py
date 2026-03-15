import pandas as pd
import joblib
import os

def run_batch_prediction():
    print("🚀 Chargement des modèles et des données de test...")
    
    # 1. Chargement des fichiers nécessaires
    try:
        X_test = pd.read_csv('data/train_test/X_test.csv')
        # On charge aussi les vrais labels pour comparaison
        y_test_actual = pd.read_csv('data/train_test/y_test.csv')
        y_test_reg_actual = pd.read_csv('data/train_test/y_reg_test.csv')
        
        # Chargement des modèles sauvegardés par train_model.py
        clf_model = joblib.load('models/best_model.pkl')
        reg_model = joblib.load('models/regression_model.pkl')
        # Note: Le clustering utilise les données scalées, mais ici on teste sur PCA
        # Si on veut le segment, on l'ajoute comme information
    except FileNotFoundError as e:
        print(f"❌ Erreur : Fichiers manquants. {e}")
        return

    # 2. Prédictions groupées
    print("🔮 Génération des prédictions sur le dataset de test...")
    
    # Prédiction du Churn (Classification)
    churn_predictions = clf_model.predict(X_test)
    churn_probabilities = clf_model.predict_proba(X_test)[:, 1]
    
    # Prédiction des dépenses (Régression)
    spending_predictions = reg_model.predict(X_test)

    # 3. Création du DataFrame de résultats
    results = pd.DataFrame({
        'Actual_Churn': y_test_actual.values.ravel(),
        'Predicted_Churn': churn_predictions,
        'Churn_Probability': churn_probabilities.round(4),
        'Actual_Spending': y_test_reg_actual.values.ravel(),
        'Predicted_Spending': spending_predictions.round(2)
    })

    # Calcul de l'erreur de prédiction pour la régression
    results['Spending_Error'] = (results['Actual_Spending'] - results['Predicted_Spending']).abs()

    # 4. Sauvegarde des résultats
    os.makedirs('data/results', exist_ok=True)
    output_path = 'data/results/test_predictions_final.csv'
    results.to_csv(output_path, index=False)
    
    print(f"\n✅ Analyse terminée !")
    print(f"📁 Fichier sauvegardé sous : {output_path}")
    print("\n--- Aperçu des 5 premières lignes ---")
    print(results.head())

    # 5. Petit résumé rapide
    correct_churn = (results['Actual_Churn'] == results['Predicted_Churn']).sum()
    print(f"\n📈 Précision sur ce fichier : {correct_churn/len(results)*100:.2f}%")

if __name__ == "__main__":
    run_batch_prediction()
# machine-Learning
# 📊 Retail Customer Analysis & Churn Prediction

## 🚀 Objectif
Ce projet vise à analyser les comportements d'achat pour :
1.  **Segmenter la clientèle** (Clustering) afin d'identifier les profils types.
2.  **Prédire le risque de départ** (Churn Prediction) pour anticiper les pertes de clients.

---

## 🛠️ État d'avancement : Prétraitement
Le pipeline de données (`src/preprocessing.py`) est finalisé et validé avec les étapes suivantes :

* **Nettoyage & Dates** : Harmonisation des formats hétérogènes et extraction de variables temporelles (`RegYear`, `RegMonth`, `RegWeekday`).
* **Imputation Intelligente** :
    * **Médiane** : Utilisée pour l'âge et les fréquences d'achat (données asymétriques).
    * **Moyenne** : Appliquée aux scores de support technique.
    * **Mode** : Utilisé pour les variables catégorielles (Genre, Statut du compte).
* **Feature Engineering** : Création de ratios métiers stratégiques :
    * `MonetaryPerDay` : Valeur générée par jour d'ancienneté.
    * `AvgBasketValue` : Montant moyen dépensé par transaction.
* **Normalisation** : Application du `StandardScaler` (Moyenne ≈ 0, Écart-type ≈ 1) pour optimiser les algorithmes basés sur la distance comme **K-Means**.

---

## 📂 Structure des fichiers générés

À l'issue du prétraitement, les données sont organisées ainsi :

* `data/processed/processed_data.csv` : Dataset complet, nettoyé et prêt pour l'analyse globale.
* `data/train_test/` : Données splittées en **80/20** avec stratification sur la cible (Churn) :
    * `X_train.csv` / `y_train.csv` : Données d'entraînement.
    * `X_test.csv` / `y_test.csv` : Données de test pour la validation finale.

---
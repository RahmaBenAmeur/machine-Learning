
# 📊 Retail Intelligence : Segmentation, Churn & Spending Prediction

## 🚀 Vision Business & Objectifs
Dans le secteur du e-commerce de cadeaux, la donnée est le levier principal de la personnalisation. Ce projet répond à trois problématiques stratégiques pour optimiser la relation client :

1. **La Segmentation (Clustering) :** *"Qui sont mes clients ?"* Identifier les profils types (acheteurs occasionnels vs VIP) pour personnaliser les campagnes marketing.
2. **La Classification (Churn) :** *"Qui risque de partir ?"* Anticiper le départ des clients. Il est **5 à 10 fois plus coûteux** d'acquérir un nouveau client que de retenir un client existant.
3. **La Régression (Spending) :** *"Quel est le potentiel financier ?"* Estimer la valeur monétaire future pour prioriser les actions commerciales sur les clients à forte valeur ajoutée.

> **Synergie des modèles :** Nous segmentons pour savoir **à qui parler**, nous prédisons le churn pour savoir **qui sauver**, et nous estimons les dépenses pour savoir **quel budget investir**.

---

## 🏗️ Architecture du Pipeline ML

### 1. 🛠️ Prétraitement & Feature Engineering (`src/preprocessing.py`)
Le fondement du projet repose sur une préparation rigoureuse des données :
* **Nettoyage & Imputation :** Traitement des valeurs manquantes par méthodes statistiques (Médiane pour l'âge, Mode pour le genre).
* **Ingénierie de variables :** Création de ratios métiers comme `AvgBasketValue` (panier moyen) et `MonetaryPerDay`.
* **Réduction de Dimension (PCA) :** Compression en **10 composantes principales** (PC1 à PC10), conservant **90% de la variance** pour éliminer le bruit tout en optimisant les calculs.

### 2. 🏋️ Entraînement & Optimisation (`src/train_model.py`)
* **Équilibrage (SMOTE) :** Application du sur-échantillonnage synthétique pour corriger le déséquilibre des classes et améliorer la détection des départs.
* **Benchmark Multimodèle :** Comparaison de KNN, Decision Tree, Random Forest et XGBoost.
* **Modèle Champion :** Le **Random Forest** a été sélectionné pour sa robustesse et sa précision exceptionnelle.

### 3. 🔮 Inférence & Validation (`src/predict.py`)
Validation sur un jeu de données "test" (données jamais vues par le modèle) pour garantir la fiabilité des prédictions en conditions réelles.

---

## 📈 Performances Finales (Modèle de Production)
Le modèle de classification atteint des scores très satisfaisants :

| Métrique | Score | Signification |
| :--- | :--- | :--- |
| **Accuracy** | **90.15%** | Précision globale du modèle sur le jeu de test. |
| **Precision** | **87.41%** | Fiabilité des alertes de Churn (peu de faux positifs). |
| **Recall** | **83.39%** | Capacité à détecter les clients qui vont réellement partir. |
| **ROC-AUC** | **0.9449** | Excellente capacité de séparation des classes. |

---

## 📂 Organisation du Projet
```plaintext
├── data/
│   ├── processed/          # Données nettoyées et transformées
│   ├── train_test/         # Splits X/y pour l'entraînement
│   └── results/            # Prédictions batch finalisées
├── models/
│   ├── best_model.pkl      # Modèle de Classification (Churn)
│   ├── regression_model.pkl# Modèle de Régression (Dépenses)
│   ├── scaler.pkl          # Normalisation sauvegardée
│   └── pca_model.pkl       # Transformation PCA sauvegardée
└── src/
    ├── preprocessing.py    # Pipeline de préparation
    ├── train_model.py      # Script d'entraînement
    ├── predict.py          # Script de test et d'inférence
    └── utils.py            # Fonctions utilitaires et graphiques
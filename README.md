# 📊 Retail Intelligence : Segmentation, Churn & Spending Prediction

## 🚀 Vision Business & Objectifs
Dans le secteur du e-commerce de cadeaux, la donnée est le levier principal de la personnalisation. Ce projet répond à trois problématiques stratégiques pour optimiser la relation client :

1. **La Segmentation (Clustering) :** *"Qui sont mes clients ?"* Identifier les profils types (VIP, Occasionnels, A risque) pour personnaliser les campagnes marketing.
2. **La Classification (Churn) :** *"Qui risque de partir ?"* Anticiper le départ des clients pour mettre en place des actions de rétention ciblées.
3. **La Régression (Spending) :** *"Quel est le potentiel financier ?"* Estimer la valeur monétaire future pour prioriser les investissements sur les clients à forte valeur ajoutée.

> **Synergie des modèles :** Nous segmentons pour savoir **à qui parler**, nous prédisons le churn pour savoir **qui sauver**, et nous estimons les dépenses pour savoir **quel budget investir**.

---

## 🏗️ Architecture du Pipeline ML

### 1. 🛠️ Prétraitement & Feature Engineering (`src/preprocessing.py`)
Le fondement du projet repose sur une préparation rigoureuse des données :
* **Nettoyage & Imputation :** Traitement des valeurs manquantes et suppression des outliers via **Isolation Forest**.
* **Réduction de Dimension (PCA) :** Compression en **10 composantes principales**, conservant l'essentiel de la variance pour éliminer le bruit.
* **Clustering Intégré :** Entraînement d'un modèle **K-Means** directement sur les composantes PCA pour une segmentation comportementale précise.

### 2. 🏋️ Entraînement & Optimisation (`src/train_model.py`)
* **Équilibrage (SMOTE) :** Application du sur-échantillonnage synthétique pour corriger le déséquilibre des classes Churn.
* **Benchmark Multimodèle :** Comparaison de KNN, Decision Tree, Random Forest et XGBoost.
* **Modèle Champion :** Le modèle **XGBoost** a été sélectionné comme meilleur classifieur pour sa performance supérieure en généralisation.

### 3. 🔮 Inférence & Validation (`src/predict.py`)
Un script d'inférence complet qui charge les modèles sauvegardés et génère un rapport consolidé (Segment + Risque Churn + Dépense prévue) pour chaque client.

---

## 📈 Performances Finales (Modèle de Production)
Le modèle de classification (XGBoost) affiche des résultats robustes sur le jeu de test :

| Métrique | Score | Signification |
| :--- | :--- | :--- |
| **Accuracy** | **88.81%** | Précision globale des prédictions. |
| **Precision** | **83.51%** | Fiabilité des alertes de Churn (peu de faux positifs). |
| **Recall** | **84.10%** | Capacité à détecter les clients qui vont réellement partir. |
| **ROC-AUC** | **0.9449** | Excellente capacité de séparation des classes. |

---

## 📂 Organisation du Projet
```plaintext
├── data/
│   ├── raw/                # Dataset original
│   ├── processed/          # Données complètes après PCA et Clustering
│   ├── train_test/         # Splits (X_train, y_train, etc.) pour les modèles
│   └── results/            # Rappels d'inférence (test_predictions_complet.csv)
├── models/
│   ├── best_model.pkl       # Modèle XGBoost (Classification Churn)
│   ├── regression_model.pkl # Modèle Random Forest (Dépenses)
│   ├── kmeans_model.pkl     # Modèle K-Means (Segmentation)
│   ├── scaler.pkl           # Normalisation des données
│   └── pca_model.pkl        # Transformation PCA
├── reports/                # Graphiques, matrices de confusion et courbes ROC
└── src/
    ├── preprocessing.py    # Pipeline de préparation et clustering
    ├── train_model.py      # Script d'entraînement et benchmark
    ├── predict.py          # Script de test final et inférence
    └── utils.py            # Fonctions de calcul de métriques et plots
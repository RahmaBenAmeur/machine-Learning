# 📊 Retail Intelligence : Segmentation, Churn & Spending Prediction

## 🚀 Vision Business & Objectifs
Dans le secteur du e-commerce, la donnée est le levier principal de la personnalisation. Ce projet répond à trois problématiques stratégiques pour optimiser la relation client :

1. **La Segmentation (Clustering) :** *"Qui sont mes clients ?"* Identifier les profils types (VIP, Acheteurs de Gros Volume, Standards, À risque) pour personnaliser les campagnes marketing.
2. **La Classification (Churn) :** *"Qui risque de partir ?"* Anticiper le départ des clients pour mettre en place des actions de rétention ciblées.
3. **La Régression (Spending) :** *"Quel est le potentiel financier ?"* Estimer la valeur monétaire future (**MonetaryTotal**) pour prioriser les investissements sur les clients à forte valeur ajoutée.

> **Synergie des modèles :** Nous segmentons pour savoir **à qui parler**, nous prédisons le churn pour savoir **qui sauver**, et nous estimons les dépenses pour savoir **quel budget investir**.

---

## 🏗️ Architecture du Pipeline ML

### 1. 🛠️ Prétraitement & Feature Engineering (`src/preprocessing.py`)
Le fondement du projet repose sur une préparation rigoureuse des données :
* **Nettoyage & Outliers :** Suppression des anomalies via **Isolation Forest** (contamination 0.06) et imputation par la médiane.
* **Feature Engineering :** Extraction temporelle (`RegYear`, `RegMonth`) et encodage automatique de 13 variables textuelles.
* **Réduction de Dimension (PCA) :** Compression en **10 composantes principales**, conservant ~90% de la variance pour éliminer le bruit.
* **Clustering :** Modèle **K-Means** entraîné sur les composantes PCA pour définir 4 segments comportementaux précis.

### 2. 🏋️ Entraînement & Optimisation (`src/train_model.py`)
* **Équilibrage (SMOTE) :** Application du sur-échantillonnage synthétique pour corriger le déséquilibre des classes de Churn.
* **Benchmark Multimodèle :** Comparaison rigoureuse de KNN, Decision Tree, Random Forest et XGBoost.
* **Modèle Champion :** Le **Random Forest** a été sélectionné pour sa robustesse et son excellent **F1-Score**.
* **Régression :** Utilisation d'un **Random Forest Regressor** optimisé pour minimiser la **RMSE**.

### 3. 💻 Déploiement & Interface (`app.py`)
Une application interactive développée avec **Streamlit** offrant deux modes :
* **Analyse par Lot :** Import de fichiers CSV pour un traitement massif et export des résultats.
* **Analyse Individuelle :** Formulaire dynamique traitant **40 variables features** en temps réel pour une aide à la décision immédiate.

---

## 📈 Performances du Modèle de Production (Random Forest)
Le modèle de classification affiche des résultats très robustes :

| Métrique | Score | Signification |
| :--- | :--- | :--- |
| **Accuracy** | **~89%** | Précision globale des prédictions. |
| **F1-Score** | **0.84** | Équilibre optimal entre précision et rappel. |
| **ROC-AUC** | **0.94** | Excellente capacité de séparation entre clients fidèles et Churn. |
| **RMSE (Régression)** | **Indicateur de Réf.** | Minimisation de l'écart moyen en Dinars (DT) pour les dépenses. |

---

## 📂 Organisation du Projet
```plaintext
├── data/
│   ├── raw/                # Dataset original (retail_customers_COMPLETE.csv)
│   ├── processed/          # Données après PCA et Clustering
│   └── results/            # Prédictions finales exportées
├── models/
│   ├── best_model.pkl      # Random Forest Classifier (Churn)
│   ├── regression_model.pkl # Random Forest Regressor (Dépenses)
│   ├── kmeans_model.pkl    # Modèle K-Means (Segments)
│   ├── pca_model.pkl       # Transformateur PCA (10 composantes)
│   └── scaler.pkl          # Standardisation des données
├── src/
│   ├── preprocessing.py    # Nettoyage, PCA et Clustering
│   ├── train_model.py      # Entraînement, SMOTE et Benchmark
│   ├── predict.py          # Script d'inférence (test des modèles)
│   └── utils.py            # Fonctions utilitaires (plots, métriques)
├── app.py                  # Interface interactive Streamlit
└── requirements.txt        # Dépendances (scikit-learn, pandas, streamlit, etc.)

## 🛠️ Installation et Utilisation

1. **Installation des dépendances :**

```bash
pip install -r requirements.txt

2. **Entraînement des modèles :**

```bash
python src/train_model.py

2. **Lancement de l'interface Streamlit :**

```bash
streamlit run app/app.py

Auteur : Rahma Ben Ameur – Engineering Student at ENIS (GI2).
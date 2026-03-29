import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Retail Intelligence GI2", layout="wide")

cluster_labels = {
    0: "Ambassadeurs (Fidèles & VIP)",
    1: "Acheteurs de Gros Volume",
    2: "Clients Actifs Standards",
    3: "Nouveaux / À Risque"
}

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    # Chargement des fichiers .pkl
    clf = joblib.load('models/best_model_churn.pkl')      
    reg = joblib.load('models/regression_model_raw.pkl') 
    kmeans = joblib.load('models/kmeans_model.pkl')      
    sc = joblib.load('models/scaler.pkl')                
    pca_mod = joblib.load('models/pca_model.pkl')        
    
    # Récupération des noms de colonnes exacts attendus par chaque étape
    cols_scaler = sc.feature_names_in_.tolist()
    cols_regres = reg.feature_names_in_.tolist()
    
    return clf, reg, kmeans, sc, pca_mod, cols_scaler, cols_regres

clf, reg, kmeans, sc, pca_mod, cols_scaler, cols_reg = load_resources()

# --- 3. FONCTION D'ALIGNEMENT DES DONNÉES ---
def align_features(df_input, target_columns):
    df_res = df_input.copy()
    
    # Encodage automatique des colonnes textes si présentes
    for col in df_res.select_dtypes(include=['object']).columns:
        df_res[col] = pd.Categorical(df_res[col]).codes
        
    # Création des colonnes manquantes (remplies par 0)
    for col in target_columns:
        if col not in df_res.columns:
            df_res[col] = 0
            
    # On retourne uniquement les colonnes attendues dans l'ordre exact
    return df_res[target_columns].fillna(0)

# --- 4. INTERFACE UTILISATEUR ---
st.title(" Retail Intelligence : Système de Prédiction 360°")
st.sidebar.header(" Importation")
uploaded_file = st.sidebar.file_uploader("Importer le fichier CSV", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"Fichier chargé : {len(df_raw)} lignes détectées.")

    try:
        # --- ÉTAPE A : PRÉPARATION, PCA ET CLUSTERING ---
        # 1. Aligner pour le scaler (ex: 40 features)
        df_for_scaler = align_features(df_raw, cols_scaler)
        X_scaled = sc.transform(df_for_scaler)
        
        # 2. Appliquer la PCA
        X_pca = pca_mod.transform(X_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])
        
        # 3. Prédictions Churn et Cluster
        df_raw['Churn_Pred'] = clf.predict(X_pca_df)
        df_raw['Cluster_ID'] = kmeans.predict(X_pca_df)
        df_raw['Segment'] = df_raw['Cluster_ID'].map(cluster_labels)

        # --- ÉTAPE B : RÉGRESSION (Dépenses prévues) ---
        # Aligner spécifiquement pour la régression (ex: 31 features)
        df_for_reg = align_features(df_raw, cols_reg)
        preds_raw = reg.predict(df_for_reg)
        
        # Si votre modèle a été entraîné sur le log des ventes, on inverse
        df_raw['Depense_Prevue_DT'] = np.expm1(preds_raw).round(2)

        # --- ÉTAPE C : AFFICHAGE DES RÉSULTATS ---
        st.subheader(" Résultats des Prédictions")
        # On affiche les colonnes clés en premier
        cols_to_show = ['Segment', 'Churn_Pred', 'Depense_Prevue_DT']
        other_cols = [c for c in df_raw.columns if c not in cols_to_show]
        st.dataframe(df_raw[cols_to_show + other_cols])

        # --- ÉTAPE D : ANALYSE SCIENTIFIQUE DES CLUSTERS (Section conservée) ---
        st.markdown("---")
        st.subheader(" Analyse des Moyennes pour l'interprétation")
        
        # Calcul des moyennes par Cluster sur toutes les colonnes numériques
        df_analyse = df_raw.groupby('Cluster_ID').mean(numeric_only=True)
        
        # Affichage avec mise en évidence des maximums en vert
        st.dataframe(df_analyse.style.highlight_max(axis=0, color='#2E7D32'))
        
        st.info("""
        ** Guide d'interprétation pour ton rapport :**
        - **Dépense_Prevue_DT max** : Tes clients les plus rentables (**VIP**).
        - **Frequency max** : Tes clients les plus réguliers (**Fidèles**).
        - **AvgQuantityPerTransaction max** : Clients qui achètent en masse (**Volume**).
        - **CustomerTenureDays min** : Clients inscrits récemment (**Nouveaux**).
        - **Churn_Pred proche de 1** : Clients avec une forte probabilité de départ (**À Risque**).
        """)

    except Exception as e:
        # Affichage de l'erreur si les dimensions ou colonnes ne correspondent toujours pas
        st.error(f"Erreur technique lors du calcul : {e}")

else:
    st.info("Veuillez importer un fichier CSV pour commencer l'analyse.")

st.caption("Déploiement Retail Intelligence | Rahma Ben Ameur - ENIS GI2 S3")
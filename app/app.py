import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION & ÉTIQUETTES ---
st.set_page_config(page_title="Retail Intelligence GI2", layout="wide")

# Labels mis à jour selon tes dernières analyses de moyennes
cluster_labels = {
    0: "Ambassadeurs (Fidèles & VIP)",
    1: "Acheteurs de Gros Volume",
    2: "Clients Actifs Standards",
    3: "Nouveaux / À Risque"
}

# --- 2. CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_resources():
    clf = joblib.load('models/best_model.pkl')      # Classification
    reg = joblib.load('models/regression_model.pkl') # Régression
    kmeans = joblib.load('models/kmeans_model.pkl')  # Clustering
    sc = joblib.load('models/scaler.pkl')           # Scaler (40 features)
    pca_mod = joblib.load('models/pca_model.pkl')    # PCA (10 comp)
    feature_names = sc.feature_names_in_.tolist()
    return clf, reg, kmeans, sc, pca_mod, feature_names

clf, reg, kmeans, sc, pca_mod, feature_names = load_resources()

# --- 3. INTERFACE UTILISATEUR ---
st.title("📊 Retail Intelligence : Système de Prédiction 360°")
st.sidebar.header("📂 Validation par Lot")
uploaded_file = st.sidebar.file_uploader("Importer X_test_brut_40.csv", type="csv")

# --- 4. TRAITEMENT DU FICHIER CSV ---
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    
    if df_test.shape[1] == 40:
        st.success(f"Fichier chargé : {len(df_test)} lignes détectées.")
        
        try:
            # --- NETTOYAGE DES DONNÉES (Anti-NaN et Anti-String) ---
            df_numeric = df_test.copy()
            
            # Conversion du texte en codes numériques
            for col in df_numeric.select_dtypes(include=['object']).columns:
                df_numeric[col] = pd.Categorical(df_numeric[col]).codes
            
            # Remplissage des valeurs vides par 0 pour éviter l'erreur PCA
            df_numeric = df_numeric.fillna(0)
            
            # --- PIPELINE MATHÉMATIQUE ---
            data_scaled = sc.transform(df_numeric.values) 
            data_pca = pca_mod.transform(data_scaled)
            X_final = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(10)])
            
            # --- EXÉCUTION DES MODÈLES ---
            df_test['Churn_Pred'] = clf.predict(X_final)
            df_test['Cluster_ID'] = kmeans.predict(X_final.values)
            df_test['Depense_Prevue_DT'] = reg.predict(X_final).round(2)
            
            # Application des noms de segments
            df_test['Segment'] = df_test['Cluster_ID'].map(cluster_labels)
            
            # --- AFFICHAGE DES RÉSULTATS ---
            st.write("### 📋 Tableau de Bord des Prédictions")
            cols_order = ['Segment', 'Churn_Pred', 'Depense_Prevue_DT'] + [c for c in df_test.columns if c not in ['Segment', 'Churn_Pred', 'Depense_Prevue_DT', 'Cluster_ID']]
            st.dataframe(df_test[cols_order])

            # --- ANALYSE SCIENTIFIQUE DES CLUSTERS ---
            st.markdown("---")
            st.subheader("🔍 Analyse des Moyennes pour l'interprétation")
            
            df_analyse = df_test.groupby('Cluster_ID').mean(numeric_only=True)
            st.dataframe(df_analyse.style.highlight_max(axis=0, color='#2E7D32'))
            
            st.info("""
            **💡 Guide d'interprétation pour ton rapport :**
            - **Dépense_Prevue_DT max** : Tes clients les plus rentables (**VIP**).
            - **Frequency max** : Tes clients les plus réguliers (**Fidèles**).
            - **AvgQuantityPerTransaction max** : Clients qui achètent en masse (**Volume**).
            - **CustomerTenureDays min** : Clients inscrits récemment (**Nouveaux**).
            - **Churn_Pred proche de 1** : Clients avec une forte probabilité de départ (**À Risque**).
            """)

        except Exception as e:
            st.error(f"Erreur technique lors du calcul : {e}")
    else:
        st.error(f"Le fichier doit avoir 40 colonnes. Reçu : {df_test.shape[1]}")

else:
    # --- 5. MODE MANUEL (CORRIGÉ POUR TEXTE & CHIFFRES) ---
    st.subheader("👤 Analyse d'un Profil Client")
    st.info("Saisissez les informations du client (Texte ou Nombre).")

    with st.form("manual_form"):
        ui_cols = st.columns(4)
        inputs = {}
        
        for i, col in enumerate(feature_names):
            with ui_cols[i % 4]:
                # On définit ici les colonnes qui sont textuelles dans ton dataset
                text_cols = ['Country', 'Gender', 'AgeCategory', 'CustomerType', 
                             'FavoriteSeason', 'PreferredTimeOfDay', 'Region', 
                             'LoyaltyLevel', 'WeekendPreference', 'BasketSizeCategory', 
                             'ProductDiversity', 'SizeCategory', 'SpendingCategory']
                
                if col in text_cols:
                    inputs[col] = st.text_input(col, value="Unknown")
                else:
                    inputs[col] = st.number_input(col, value=0.0)
        
        if st.form_submit_button("🚀 Lancer l'Analyse"):
            try:
                # 1. Création du DataFrame
                df_in = pd.DataFrame([inputs])[feature_names]
                
                # 2. Encodage automatique du texte en nombres (comme pour le CSV)
                for col in df_in.select_dtypes(include=['object']).columns:
                    df_in[col] = pd.Categorical(df_in[col]).codes
                
                # 3. Pipeline mathématique
                d_scaled = sc.transform(df_in.values)
                d_pca = pca_mod.transform(d_scaled)
                X_in = pd.DataFrame(d_pca, columns=[f'PC{i+1}' for i in range(10)])
                
                # 4. Prédictions
                res_churn = clf.predict(X_in)[0]
                res_clust_id = kmeans.predict(X_in.values)[0]
                res_spend = reg.predict(X_in)[0]
                
                nom_profil = cluster_labels.get(res_clust_id, "Client Standard")

                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Statut Fidélité", "🔴 Risque de départ" if res_churn == 1 else "🟢 Fidèle")
                c2.metric("Profil Client", nom_profil) 
                c3.metric("Potentiel d'achat", f"{res_spend:.2f} DT")
                
                st.success("Analyse individuelle terminée.")

            except Exception as e:
                st.error(f"Erreur lors du traitement des données : {e}")

# --- PIED DE PAGE ---
st.caption("Déploiement Retail Intelligence | Rahma Ben Ameur - ENIS GI2 S3")
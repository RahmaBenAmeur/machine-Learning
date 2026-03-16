import pandas as pd
import joblib
import os

# --- 1. CONFIGURATION DES CHEMINS ---
# REMPLACEZ 'dataset.csv' PAR LE NOM RÉEL DE VOTRE FICHIER DANS data/raw/
RAW_DATA_PATH = 'data/raw/retail_customers_COMPLETE_CATEGORICAL.csv' 
SCALER_PATH = 'models/scaler.pkl'
X_TEST_PCA_PATH = 'data/train_test/X_test.csv'
OUTPUT_PATH = 'data/train_test/X_test_brut_40.csv'

# --- 2. VÉRIFICATION DU FICHIER ---
if not os.path.exists(RAW_DATA_PATH):
    print(f"❌ Erreur : Le fichier '{RAW_DATA_PATH}' est introuvable.")
    print("👉 Allez dans votre dossier 'data/raw/' et copiez ici le nom exact du fichier .csv")
    exit()

# --- 3. CHARGEMENT ---
print("⏳ Chargement des données...")
df_raw = pd.read_csv(RAW_DATA_PATH)
sc = joblib.load(SCALER_PATH)
colonnes_attendues = sc.feature_names_in_.tolist() # Les 40 colonnes

# --- 4. CRÉATION DES COLONNES MANQUANTES ---
# On s'assure que RegYear et RegMonth existent
if 'RegYear' not in df_raw.columns:
    df_raw['RegYear'] = 2024 # Valeur par défaut
if 'RegMonth' not in df_raw.columns:
    df_raw['RegMonth'] = 1

# On vérifie s'il manque d'autres colonnes parmi les 40
for col in colonnes_attendues:
    if col not in df_raw.columns:
        df_raw[col] = 0

# --- 5. EXTRACTION DU TEST SET ---
# On récupère le nombre de lignes du X_test actuel (PC1...PC10)
df_pc_test = pd.read_csv(X_TEST_PCA_PATH)
test_size = len(df_pc_test)

# On prend les 'test_size' dernières lignes pour correspondre au split
df_test_brut = df_raw.tail(test_size)[colonnes_attendues]

# --- 6. SAUVEGARDE ---
df_test_brut.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Succès ! Fichier généré : {OUTPUT_PATH}")
print(f"📊 Dimensions : {df_test_brut.shape} (Doit être 40 colonnes)")
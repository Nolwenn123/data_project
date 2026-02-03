import pandas as pd
import numpy as np
from pathlib import Path

# === CHEMINS ===
RAW_PATH = Path(__file__).parent.parent / "raw"
PROCESSED_PATH = Path(__file__).parent

def load_data():
    """Charge les données brutes."""
    print("📂 Chargement des données...")
    df_visits = pd.read_csv(RAW_PATH / "ed_visits_pitie_salpetriere_synthetic.csv", 
                            parse_dates=['arrival_datetime'])
    df_context = pd.read_csv(RAW_PATH / "daily_context.csv", parse_dates=['date'])
    
    print(f"   Visites: {len(df_visits):,} lignes")
    print(f"   Contexte: {len(df_context):,} jours")
    return df_visits, df_context


def create_features(df):
    """Crée les features temporelles et dérivées."""
    print("⚙️  Feature engineering...")
    
    # Features temporelles
    df['date'] = df['arrival_datetime'].dt.date.astype('datetime64[ns]')
    df['hour'] = df['arrival_datetime'].dt.hour
    df['day_of_week'] = df['arrival_datetime'].dt.dayofweek
    df['month'] = df['arrival_datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 20) | (df['hour'] < 8)).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2, 3]).astype(int)
    
    # Score épidémique combiné
    epidemic_cols = ['flu_level', 'bronchiolitis_level', 'covid_level']
    if all(col in df.columns for col in epidemic_cols):
        df['epidemic_score'] = df[epidemic_cols].sum(axis=1)
    
    # Taux d'occupation (normalisé)
    if 'beds_occupied' in df.columns:
        df['occupancy_rate'] = df['beds_occupied'] / df['beds_occupied'].max()
    
    # Groupes d'âge
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], 
                                  bins=[0, 18, 40, 65, 120], 
                                  labels=['child', 'adult', 'middle', 'elderly'])
    
    # Triage critique
    if 'triage_level' in df.columns:
        df['is_critical'] = (df['triage_level'] <= 2).astype(int)
    
    return df


def encode_categoricals(df):
    """Encode les variables catégorielles."""
    print("🔢 Encodage des catégories...")
    
    # Colonnes à encoder en one-hot
    cat_cols = ['arrival_mode', 'ed_disposition', 'age_group']
    
    for col in cat_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
            print(f"   ✓ {col} encodé")
    
    return df


def handle_missing(df):
    """Gère les valeurs manquantes."""
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        print(f"🔧 Gestion des NaN ({missing_before} valeurs)...")
        
        # Numériques → médiane
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Catégorielles → mode ou 'Unknown'
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
        
        print(f"   ✓ {missing_before} NaN traités")
    else:
        print("✓ Pas de valeurs manquantes")
    
    return df


def clean_columns(df):
    """Supprime les colonnes inutiles pour le ML."""
    print("🗑️  Nettoyage des colonnes...")
    
    cols_to_drop = [
        'visit_id',           # Identifiant unique
        'arrival_datetime',   # Déjà extrait en features
        'date',               # Déjà extrait
        'chief_complaint',    # Texte libre (à traiter séparément si besoin)
    ]
    
    dropped = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"   ✓ Supprimé: {dropped}")
    
    return df


def save_data(df, filename="ed_data_processed.csv"):
    """Sauvegarde les données traitées."""
    output_path = PROCESSED_PATH / filename
    df.to_csv(output_path, index=False)
    print(f"\n💾 Sauvegardé: {output_path}")
    print(f"   → {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    return output_path


def main():
    """Pipeline principal de preprocessing."""
    print("=" * 50)
    print("PREPROCESSING - ED Data")
    print("=" * 50 + "\n")
    
    # 1. Charger
    df_visits, df_context = load_data()
    
    # 2. Vérifier si fusion nécessaire
    context_cols = ['beds_occupied', 'flu_level', 'heatwave']
    need_merge = not all(col in df_visits.columns for col in context_cols)
    
    if need_merge:
        print("🔗 Fusion avec contexte journalier...")
        df_visits['date'] = df_visits['arrival_datetime'].dt.date.astype('datetime64[ns]')
        df = df_visits.merge(df_context, on='date', how='left')
    else:
        print("✓ Contexte déjà présent dans les visites")
        df = df_visits.copy()
    
    # 3. Pipeline
    df = create_features(df)
    df = handle_missing(df)
    df = encode_categoricals(df)
    df = clean_columns(df)
    
    # 4. Sauvegarder
    save_data(df)
    
    # 5. Résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ")
    print("=" * 50)
    print(f"Features numériques: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Colonnes finales: {list(df.columns)[:8]}...")
    
    return df


if __name__ == "__main__":
    df = main()

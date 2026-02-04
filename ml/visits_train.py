"""
Modèle de prédiction du nombre de visites aux urgences (J+1)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

# === CHEMINS ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = Path(__file__).parent


def load_data():
    """Charge et prépare les données."""
    print("📂 Chargement des données...")
    
    df_context = pd.read_csv(DATA_PATH / "daily_context.csv", parse_dates=['date'])
    df_visits = pd.read_csv(DATA_PATH / "ed_visits_pitie_salpetriere_synthetic.csv", 
                             parse_dates=['arrival_datetime'])
    
    # Compter les visites par jour
    df_visits['date'] = pd.to_datetime(df_visits['arrival_datetime'].dt.date)
    daily_visits = df_visits.groupby('date').size().reset_index(name='visits_count')
    
    # S'assurer que les dates sont au même format
    df_context['date'] = pd.to_datetime(df_context['date'])
    daily_visits['date'] = pd.to_datetime(daily_visits['date'])
    
    # Fusionner
    df = df_context.merge(daily_visits, on='date', how='left')
    
    # Vérifier si la colonne existe après merge
    if 'visits_count' not in df.columns:
        # Si pas de merge, calculer directement
        print("   ⚠️  Merge échoué, calcul direct...")
        visits_per_day = df_visits.groupby('date').size()
        df['visits_count'] = df['date'].map(visits_per_day).fillna(0)
    else:
        df['visits_count'] = df['visits_count'].fillna(0)
    
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"   → {len(df)} jours chargés")
    print(f"   → Moyenne: {df['visits_count'].mean():.0f} visites/jour")
    return df


def create_features(df):
    """Crée les features pour la prédiction J+1."""
    print("⚙️  Création des features...")
    
    # Features temporelles
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2, 3]).astype(int)
    
    # Score épidémique
    df['epidemic_score'] = df['flu_level'] + df['bronchiolitis_level'] + df['covid_level']
    
    # === CIBLE : visites du LENDEMAIN ===
    df['visits_tomorrow'] = df['visits_count'].shift(-1)
    
    # Historique (features basées sur les jours précédents)
    df['visits_yesterday'] = df['visits_count'].shift(1)
    df['visits_avg_7days'] = df['visits_count'].rolling(7).mean().shift(1)
    df['visits_avg_30days'] = df['visits_count'].rolling(30).mean().shift(1)
    
    # Supprimer les lignes avec NaN (début et fin)
    df = df.dropna()
    
    print(f"   → {len(df)} jours utilisables")
    return df


def train_model(df):
    """Entraîne le modèle de prédiction."""
    
    # Features d'entrée
    features = [
        # Temporel
        'day_of_week', 'month', 'is_weekend', 'is_winter',
        # Épidémies
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        # Événements
        'heatwave', 'strike', 'mass_casualty_event',
        # Ressources
        'beds_occupied', 'icu_beds_occupied',
        'staff_doctors_available', 'staff_nurses_available',
        # Historique
        'visits_yesterday', 'visits_avg_7days', 'visits_avg_30days'
    ]
    
    X = df[features]
    y = df['visits_tomorrow']  # ← Prédire les visites de DEMAIN
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Données:")
    print(f"   Train: {len(X_train)} jours")
    print(f"   Test:  {len(X_test)} jours")
    print(f"   Moyenne visites/jour: {y.mean():.0f}")
    
    # Entraînement
    print("\n🚀 Entraînement du modèle...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1  # Utilise tous les CPU
    )
    model.fit(X_train, y_train)
    print("   ✓ Modèle entraîné")
    
    # Évaluation
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Performance:")
    print(f"   MAE  = {mae:.1f} visites (erreur moyenne)")
    print(f"   RMSE = {rmse:.1f} visites")
    print(f"   R²   = {r2:.1%} (qualité du modèle)")
    
    # Importance des features
    print("\n🔍 Features les plus importantes:")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.1%}")
    
    return model, features


def save_model(model, features):
    """Sauvegarde le modèle et les métadonnées."""
    
    # Sauvegarder le modèle
    model_file = MODEL_PATH / "visits_predictor.joblib"
    joblib.dump(model, model_file)
    
    # Sauvegarder les features utilisées
    features_file = MODEL_PATH / "features.txt"
    with open(features_file, 'w') as f:
        f.write('\n'.join(features))
    
    print(f"\n💾 Sauvegardé:")
    print(f"   → {model_file}")
    print(f"   → {features_file}")


def main():
    """Pipeline principal."""
    print("=" * 50)
    print("ENTRAÎNEMENT - Prédiction visites J+1")
    print("=" * 50 + "\n")
    
    # 1. Charger
    df = load_data()
    
    # 2. Features
    df = create_features(df)
    
    # 3. Entraîner
    model, features = train_model(df)
    
    # 4. Sauvegarder
    save_model(model, features)
    
    print("\n" + "=" * 50)
    print("✅ Modèle prêt à l'emploi !")
    print("=" * 50)


if __name__ == "__main__":
    main()

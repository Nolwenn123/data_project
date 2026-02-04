"""
Modèle de prédiction du temps d'attente aux urgences
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = Path(__file__).parent


def main():
    print("=" * 50)
    print("ENTRAÎNEMENT - Temps d'attente")
    print("=" * 50 + "\n")
    
    # 1. Charger les données processées
    df = pd.read_csv(DATA_PATH / "ed_data_processed.csv")
    print(f"📂 {len(df):,} lignes chargées")
    
    # 2. Définir features et cible
    target = 'wait_time_minutes'
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_winter',
        'age', 'triage_level', 'is_critical',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'beds_occupied', 'occupancy_rate',
        'staff_doctors_available', 'staff_nurses_available',
        'heatwave', 'strike', 'mass_casualty_event'
    ]
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df[target]
    
    print(f"   → {len(features)} features")
    print(f"   → Temps moyen: {y.mean():.0f} min")
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   → Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 4. Entraîner
    print("\n🚀 Entraînement...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   ✓ Modèle entraîné")
    
    # 5. Évaluer
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Résultats:")
    print(f"   MAE = {mae:.1f} min (erreur moyenne)")
    print(f"   R²  = {r2:.1%}")
    
    # 6. Top features
    print("\n🔍 Top features:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])[:5]:
        print(f"   {feat}: {imp:.1%}")
    
    # 7. Sauvegarder
    joblib.dump(model, MODEL_PATH / "wait_time_predictor.joblib")
    print(f"\n💾 Sauvegardé: ml/wait_time_predictor.joblib")


if __name__ == "__main__":
    main()

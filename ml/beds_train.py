"""
Modèle de prédiction des lits occupés J+1
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = Path(__file__).parent


def main():
    print("=" * 50)
    print("ENTRAÎNEMENT - Lits occupés J+1")
    print("=" * 50 + "\n")
    
    # 1. Charger
    df = pd.read_csv(DATA_PATH / "daily_context.csv", parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"📂 {len(df)} jours chargés")
    
    # 2. Features temporelles
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2, 3]).astype(int)
    df['epidemic_score'] = df['flu_level'] + df['bronchiolitis_level'] + df['covid_level']
    
    # 3. Cible = lits occupés DEMAIN
    df['beds_tomorrow'] = df['beds_occupied'].shift(-1)
    
    # Historique
    df['beds_yesterday'] = df['beds_occupied'].shift(1)
    df['beds_avg_7days'] = df['beds_occupied'].rolling(7).mean().shift(1)
    
    df = df.dropna()
    
    target = 'beds_tomorrow'
    
    features = [
        'day_of_week', 'month', 'is_weekend', 'is_winter',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'heatwave', 'strike', 'mass_casualty_event',
        'beds_occupied', 'icu_beds_occupied',
        'staff_doctors_available', 'staff_nurses_available',
        'beds_yesterday', 'beds_avg_7days'
    ]
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df[target]
    
    print(f"   → {len(features)} features")
    print(f"   → Lits moyens: {y.mean():.0f}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   → Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 5. Entraîner
    print("\n🚀 Entraînement...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   ✓ Modèle entraîné")
    
    # 6. Évaluer
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Résultats:")
    print(f"   MAE = {mae:.1f} lits (erreur moyenne)")
    print(f"   R²  = {r2:.1%}")
    
    # 7. Top features
    print("\n🔍 Top features:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])[:5]:
        print(f"   {feat}: {imp:.1%}")
    
    # 8. Sauvegarder
    joblib.dump(model, MODEL_PATH / "beds_predictor.joblib")
    print(f"\n💾 Sauvegardé: ml/beds_predictor.joblib")


if __name__ == "__main__":
    main()

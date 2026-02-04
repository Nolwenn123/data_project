"""
Modèle de prédiction d'admission (classification)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = Path(__file__).parent


def main():
    print("=" * 50)
    print("ENTRAÎNEMENT - Prédiction admission")
    print("=" * 50 + "\n")
    
    # 1. Charger
    df = pd.read_csv(DATA_PATH / "ed_data_processed.csv")
    print(f"📂 {len(df):,} lignes chargées")
    
    # 2. Créer la cible (admission = pas discharge)
    # Chercher les colonnes d'admission
    admission_cols = [c for c in df.columns if 'ed_disposition' in c.lower() and 'admission' in c.lower()]
    discharge_cols = [c for c in df.columns if 'ed_disposition' in c.lower() and 'discharge' in c.lower()]
    
    if admission_cols:
        df['is_admitted'] = df[admission_cols].max(axis=1)
    elif discharge_cols:
        df['is_admitted'] = 1 - df[discharge_cols].max(axis=1)
    else:
        # Fallback: chercher la colonne originale
        print("⚠️  Colonnes admission non trouvées, tentative alternative...")
        if 'ed_disposition' in df.columns:
            df['is_admitted'] = df['ed_disposition'].str.contains('admission|admitted|hospitalized', case=False, na=False).astype(int)
        else:
            print("❌ Impossible de créer la cible admission")
            return
    
    target = 'is_admitted'
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_winter',
        'age', 'triage_level', 'is_critical',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'beds_occupied', 'occupancy_rate',
        'staff_doctors_available', 'staff_nurses_available',
        'heatwave', 'strike', 'mass_casualty_event',
        'wait_time_minutes'
    ]
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df[target]
    
    print(f"   → {len(features)} features")
    print(f"   → Taux admission: {y.mean():.1%}")
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   → Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 4. Entraîner (CLASSIFIER, pas Regressor !)
    print("\n🚀 Entraînement...")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   ✓ Modèle entraîné")
    
    # 5. Évaluer
    y_pred = model.predict(X_test)
    
    print(f"\n📈 Résultats:")
    print(f"   Accuracy  = {accuracy_score(y_test, y_pred):.1%}")
    print(f"   Precision = {precision_score(y_test, y_pred, zero_division=0):.1%}")
    print(f"   Recall    = {recall_score(y_test, y_pred, zero_division=0):.1%}")
    print(f"   F1-Score  = {f1_score(y_test, y_pred, zero_division=0):.1%}")
    
    # 6. Top features
    print("\n🔍 Top features:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])[:5]:
        print(f"   {feat}: {imp:.1%}")
    
    # 7. Sauvegarder
    joblib.dump(model, MODEL_PATH / "admission_predictor.joblib")
    print(f"\n💾 Sauvegardé: ml/admission_predictor.joblib")


if __name__ == "__main__":
    main()

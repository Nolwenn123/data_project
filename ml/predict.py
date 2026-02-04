"""
Script de prédiction - Utilise les modèles entraînés
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent


def load_models():
    """Charge tous les modèles disponibles."""
    models = {}
    
    model_files = {
        'visits': 'visits_predictor.joblib',
        'wait_time': 'wait_time_predictor.joblib',
        'admission': 'admission_predictor.joblib',
        'beds': 'beds_predictor.joblib'
    }
    
    for name, filename in model_files.items():
        filepath = MODEL_PATH / filename
        if filepath.exists():
            models[name] = joblib.load(filepath)
            print(f"✓ Modèle '{name}' chargé")
        else:
            print(f"⚠ Modèle '{name}' non trouvé")
    
    return models


def predict_visits(model, data: dict) -> float:
    """Prédit le nombre de visites pour demain."""
    features = [
        'day_of_week', 'month', 'is_weekend', 'is_winter',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'heatwave', 'strike', 'mass_casualty_event',
        'beds_occupied', 'icu_beds_occupied',
        'staff_doctors_available', 'staff_nurses_available',
        'visits_yesterday', 'visits_avg_7days', 'visits_avg_30days'
    ]
    
    X = [[data.get(f, 0) for f in features]]
    prediction = model.predict(X)[0]
    return round(prediction)


def predict_wait_time(model, data: dict) -> float:
    """Prédit le temps d'attente en minutes."""
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_winter',
        'age', 'triage_level', 'is_critical',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'beds_occupied', 'occupancy_rate',
        'staff_doctors_available', 'staff_nurses_available',
        'heatwave', 'strike', 'mass_casualty_event'
    ]
    
    X = [[data.get(f, 0) for f in features]]
    prediction = model.predict(X)[0]
    return round(prediction, 1)


def predict_admission(model, data: dict) -> dict:
    """Prédit si un patient sera admis (classification)."""
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_winter',
        'age', 'triage_level', 'is_critical',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'beds_occupied', 'occupancy_rate',
        'staff_doctors_available', 'staff_nurses_available',
        'heatwave', 'strike', 'mass_casualty_event',
        'wait_time_minutes'
    ]
    
    X = [[data.get(f, 0) for f in features]]
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    return {
        'admitted': bool(prediction),
        'probability': round(proba[1] * 100, 1)
    }


def predict_beds(model, data: dict) -> float:
    """Prédit le nombre de lits occupés demain."""
    features = [
        'day_of_week', 'month', 'is_weekend', 'is_winter',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'heatwave', 'strike', 'mass_casualty_event',
        'beds_occupied', 'icu_beds_occupied',
        'staff_doctors_available', 'staff_nurses_available',
        'beds_yesterday', 'beds_avg_7days'
    ]
    
    X = [[data.get(f, 0) for f in features]]
    prediction = model.predict(X)[0]
    return round(prediction)


# === EXEMPLE D'UTILISATION ===
if __name__ == "__main__":
    print("=" * 50)
    print("TEST DES PRÉDICTIONS")
    print("=" * 50 + "\n")
    
    # Charger les modèles
    models = load_models()
    
    # Données exemple (contexte d'un jour d'hiver avec épidémie)
    exemple_jour = {
        'day_of_week': 1,           # Mardi
        'month': 2,                 # Février
        'is_weekend': 0,
        'is_winter': 1,
        'flu_level': 3,             # Épidémie modérée
        'bronchiolitis_level': 2,
        'covid_level': 1,
        'epidemic_score': 6,
        'heatwave': 0,
        'strike': 0,
        'mass_casualty_event': 0,
        'beds_occupied': 1650,
        'icu_beds_occupied': 85,
        'staff_doctors_available': 45,
        'staff_nurses_available': 120,
        'visits_yesterday': 295,
        'visits_avg_7days': 280,
        'visits_avg_30days': 275,
        'beds_yesterday': 1640,
        'beds_avg_7days': 1620
    }
    
    # Données exemple patient
    exemple_patient = {
        **exemple_jour,
        'hour': 14,
        'is_night': 0,
        'age': 72,
        'triage_level': 3,
        'is_critical': 0,
        'occupancy_rate': 0.92,
        'wait_time_minutes': 45
    }
    
    print("\n📊 Contexte: Mardi février, épidémie modérée\n")
    
    # Prédictions
    if 'visits' in models:
        pred = predict_visits(models['visits'], exemple_jour)
        print(f"🏥 Visites prévues demain: {pred} patients")
    
    if 'beds' in models:
        pred = predict_beds(models['beds'], exemple_jour)
        print(f"🛏️  Lits occupés demain: {pred} lits")
    
    if 'wait_time' in models:
        pred = predict_wait_time(models['wait_time'], exemple_patient)
        print(f"⏱️  Temps d'attente prévu: {pred} minutes")
    
    if 'admission' in models:
        pred = predict_admission(models['admission'], exemple_patient)
        print(f"🚪 Admission probable: {'Oui' if pred['admitted'] else 'Non'} ({pred['probability']}%)")
    
    print("\n" + "=" * 50)

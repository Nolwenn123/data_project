"""
API FastAPI - Prédictions hospitalières
Hôpital Pitié-Salpêtrière
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
from pathlib import Path

# === INITIALISATION ===
app = FastAPI(
    title="API Prédictions Urgences",
    description="API pour prédire l'activité des urgences - Pitié-Salpêtrière",
    version="1.0.0"
)

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CHEMINS ===
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "ml"

# === CHARGEMENT DES MODÈLES ===
models = {}

def load_models():
    """Charge les modèles au démarrage."""
    global models
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
            print(f"⚠ Modèle '{name}' non trouvé: {filepath}")

# Charger au démarrage
load_models()


# === SCHEMAS PYDANTIC ===
class DayContext(BaseModel):
    """Contexte d'une journée pour prédiction."""
    day_of_week: int = 0          # 0=Lundi, 6=Dimanche
    month: int = 1
    is_weekend: int = 0
    is_winter: int = 0
    flu_level: float = 0
    bronchiolitis_level: float = 0
    covid_level: float = 0
    epidemic_score: Optional[float] = None
    heatwave: int = 0
    strike: int = 0
    mass_casualty_event: int = 0
    beds_occupied: int = 1500
    icu_beds_occupied: int = 80
    staff_doctors_available: int = 50
    staff_nurses_available: int = 120
    visits_yesterday: float = 280
    visits_avg_7days: float = 280
    visits_avg_30days: float = 280
    beds_yesterday: Optional[float] = None
    beds_avg_7days: Optional[float] = None


class PatientContext(BaseModel):
    """Contexte patient pour prédiction individuelle."""
    hour: int = 12
    day_of_week: int = 0
    is_weekend: int = 0
    is_night: int = 0
    is_winter: int = 0
    age: int = 50
    triage_level: int = 3
    is_critical: int = 0
    flu_level: float = 0
    bronchiolitis_level: float = 0
    covid_level: float = 0
    epidemic_score: Optional[float] = None
    beds_occupied: int = 1500
    occupancy_rate: float = 0.85
    staff_doctors_available: int = 50
    staff_nurses_available: int = 120
    heatwave: int = 0
    strike: int = 0
    mass_casualty_event: int = 0
    wait_time_minutes: Optional[float] = None


class VisitsPrediction(BaseModel):
    visits_predicted: int
    status: str


class WaitTimePrediction(BaseModel):
    wait_time_minutes: float
    status: str


class AdmissionPrediction(BaseModel):
    admitted: bool
    probability: float
    status: str


class BedsPrediction(BaseModel):
    beds_predicted: int
    status: str


# === ROUTES ===

@app.get("/")
def root():
    """Route racine."""
    return {
        "message": "API Prédictions Urgences - Pitié-Salpêtrière",
        "version": "1.0.0",
        "models_loaded": list(models.keys()),
        "endpoints": [
            "/predict/visits",
            "/predict/wait-time",
            "/predict/admission",
            "/predict/beds"
        ]
    }


@app.get("/health")
def health():
    """Vérifie que l'API fonctionne."""
    return {"status": "ok", "models_loaded": len(models)}


@app.post("/predict/visits", response_model=VisitsPrediction)
def predict_visits(context: DayContext):
    """Prédit le nombre de visites pour demain."""
    if 'visits' not in models:
        raise HTTPException(status_code=503, detail="Modèle 'visits' non disponible")
    
    # Calculer epidemic_score si non fourni
    if context.epidemic_score is None:
        context.epidemic_score = context.flu_level + context.bronchiolitis_level + context.covid_level
    
    features = [
        'day_of_week', 'month', 'is_weekend', 'is_winter',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'heatwave', 'strike', 'mass_casualty_event',
        'beds_occupied', 'icu_beds_occupied',
        'staff_doctors_available', 'staff_nurses_available',
        'visits_yesterday', 'visits_avg_7days', 'visits_avg_30days'
    ]
    
    data = context.dict()
    X = [[data.get(f, 0) for f in features]]
    
    prediction = models['visits'].predict(X)[0]
    
    return VisitsPrediction(
        visits_predicted=round(prediction),
        status="ok"
    )


@app.post("/predict/wait-time", response_model=WaitTimePrediction)
def predict_wait_time(context: PatientContext):
    """Prédit le temps d'attente pour un patient."""
    if 'wait_time' not in models:
        raise HTTPException(status_code=503, detail="Modèle 'wait_time' non disponible")
    
    if context.epidemic_score is None:
        context.epidemic_score = context.flu_level + context.bronchiolitis_level + context.covid_level
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_winter',
        'age', 'triage_level', 'is_critical',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'beds_occupied', 'occupancy_rate',
        'staff_doctors_available', 'staff_nurses_available',
        'heatwave', 'strike', 'mass_casualty_event'
    ]
    
    data = context.dict()
    X = [[data.get(f, 0) for f in features]]
    
    prediction = models['wait_time'].predict(X)[0]
    
    return WaitTimePrediction(
        wait_time_minutes=round(prediction, 1),
        status="ok"
    )


@app.post("/predict/admission", response_model=AdmissionPrediction)
def predict_admission(context: PatientContext):
    """Prédit si un patient sera admis."""
    if 'admission' not in models:
        raise HTTPException(status_code=503, detail="Modèle 'admission' non disponible")
    
    if context.epidemic_score is None:
        context.epidemic_score = context.flu_level + context.bronchiolitis_level + context.covid_level
    
    features = [
        'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_winter',
        'age', 'triage_level', 'is_critical',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'beds_occupied', 'occupancy_rate',
        'staff_doctors_available', 'staff_nurses_available',
        'heatwave', 'strike', 'mass_casualty_event',
        'wait_time_minutes'
    ]
    
    data = context.dict()
    X = [[data.get(f, 0) or 0 for f in features]]
    
    prediction = models['admission'].predict(X)[0]
    proba = models['admission'].predict_proba(X)[0]
    
    return AdmissionPrediction(
        admitted=bool(prediction),
        probability=round(proba[1] * 100, 1),
        status="ok"
    )


@app.post("/predict/beds", response_model=BedsPrediction)
def predict_beds(context: DayContext):
    """Prédit le nombre de lits occupés demain."""
    if 'beds' not in models:
        raise HTTPException(status_code=503, detail="Modèle 'beds' non disponible")
    
    if context.epidemic_score is None:
        context.epidemic_score = context.flu_level + context.bronchiolitis_level + context.covid_level
    if context.beds_yesterday is None:
        context.beds_yesterday = context.beds_occupied
    if context.beds_avg_7days is None:
        context.beds_avg_7days = context.beds_occupied
    
    features = [
        'day_of_week', 'month', 'is_weekend', 'is_winter',
        'flu_level', 'bronchiolitis_level', 'covid_level', 'epidemic_score',
        'heatwave', 'strike', 'mass_casualty_event',
        'beds_occupied', 'icu_beds_occupied',
        'staff_doctors_available', 'staff_nurses_available',
        'beds_yesterday', 'beds_avg_7days'
    ]
    
    data = context.dict()
    X = [[data.get(f, 0) for f in features]]
    
    prediction = models['beds'].predict(X)[0]
    
    return BedsPrediction(
        beds_predicted=round(prediction),
        status="ok"
    )


# === LANCEMENT ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

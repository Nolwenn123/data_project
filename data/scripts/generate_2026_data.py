"""
Script pour générer les données de 2026 (1er janvier - 5 février 2026)
Respecte les tendances saisonnières observées dans les données historiques
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

# Seed pour reproductibilité
np.random.seed(42)
random.seed(42)


def generate_daily_context_2026():
    """Génère le contexte journalier pour janvier-février 2026."""
    print("📅 Génération du contexte journalier 2026...")
    
    # Charger les données existantes pour comprendre les patterns
    df_existing = pd.read_csv(RAW_PATH / "daily_context.csv", parse_dates=['date'])
    
    # Statistiques hivernales (décembre-février des années précédentes)
    winter_data = df_existing[df_existing['date'].dt.month.isin([12, 1, 2])]
    
    # Moyennes hivernales
    winter_stats = {
        'beds_occupied': winter_data['beds_occupied'].mean(),
        'beds_std': winter_data['beds_occupied'].std(),
        'icu_beds_occupied': winter_data['icu_beds_occupied'].mean(),
        'staff_doctors_available': winter_data['staff_doctors_available'].mean(),
        'staff_nurses_available': winter_data['staff_nurses_available'].mean(),
        'staff_absence_rate': winter_data['staff_absence_rate'].mean(),
    }
    
    print(f"   Stats hivernales: lits={winter_stats['beds_occupied']:.0f}, réa={winter_stats['icu_beds_occupied']:.0f}")
    
    # Dates à générer: 1er janvier 2026 au 5 février 2026
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 2, 5)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    rows = []
    
    for date in dates:
        day_of_week = date.dayofweek
        month = date.month
        
        # Épidémies hivernales (pic en janvier)
        if month == 1:
            # Pic épidémique en janvier
            flu_level = np.clip(np.random.normal(4.0, 0.8), 2, 5)
            bronchiolitis_level = np.clip(np.random.normal(3.5, 0.7), 2, 5)
            covid_level = np.clip(np.random.normal(2.0, 0.5), 0.5, 3)
        else:  # Février
            # Légère baisse mais toujours élevé
            flu_level = np.clip(np.random.normal(3.5, 0.7), 2, 5)
            bronchiolitis_level = np.clip(np.random.normal(2.5, 0.6), 1, 4)
            covid_level = np.clip(np.random.normal(1.5, 0.4), 0.5, 2.5)
        
        # Lits occupés (plus élevé en hiver + variation jour)
        base_beds = winter_stats['beds_occupied']
        # Weekend = légèrement moins d'admissions programmées
        weekend_factor = 0.97 if day_of_week >= 5 else 1.0
        # Lundi = plus de monde (retour consultations)
        monday_factor = 1.03 if day_of_week == 0 else 1.0
        
        beds_occupied = int(np.clip(
            np.random.normal(base_beds * weekend_factor * monday_factor, winter_stats['beds_std']),
            1400, 1750
        ))
        
        # Réanimation
        icu_beds = int(np.clip(
            np.random.normal(winter_stats['icu_beds_occupied'], 8),
            70, 110
        ))
        
        # Personnel (légère baisse en hiver à cause des maladies)
        doctors = int(np.clip(np.random.normal(winter_stats['staff_doctors_available'], 5), 35, 60))
        nurses = int(np.clip(np.random.normal(winter_stats['staff_nurses_available'], 10), 100, 150))
        absence_rate = np.clip(np.random.normal(winter_stats['staff_absence_rate'] * 1.1, 0.02), 0.05, 0.20)
        
        # Événements exceptionnels (rares)
        heatwave = 0  # Pas de canicule en hiver
        strike = 1 if random.random() < 0.02 else 0  # 2% de chance
        mass_casualty = 1 if random.random() < 0.005 else 0  # 0.5% de chance
        
        # Stock EPI
        ppe_stock = int(np.clip(np.random.normal(80, 10), 50, 100))
        critical_stockout = 1 if ppe_stock < 60 else 0
        
        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'beds_total': 1800,
            'beds_occupied': beds_occupied,
            'icu_beds_total': 120,
            'icu_beds_occupied': icu_beds,
            'staff_doctors_available': doctors,
            'staff_nurses_available': nurses,
            'staff_absence_rate': round(absence_rate, 3),
            'ppe_stock_index': ppe_stock,
            'critical_stockout': critical_stockout,
            'flu_level': round(flu_level, 1),
            'bronchiolitis_level': round(bronchiolitis_level, 1),
            'covid_level': round(covid_level, 1),
            'heatwave': heatwave,
            'strike': strike,
            'mass_casualty_event': mass_casualty
        })
    
    df_new = pd.DataFrame(rows)
    print(f"   ✓ {len(df_new)} jours générés")
    return df_new


def generate_visits_2026(df_context_2026):
    """Génère les visites patients pour janvier-février 2026."""
    print("🏥 Génération des visites 2026...")
    
    # Charger données existantes pour patterns
    df_visits_existing = pd.read_csv(
        RAW_PATH / "ed_visits_pitie_salpetriere_synthetic.csv",
        parse_dates=['arrival_datetime']
    )
    
    # Stats générales
    services = df_visits_existing['service_destination'].value_counts(normalize=True).to_dict()
    arrival_modes = df_visits_existing['arrival_mode'].value_counts(normalize=True).to_dict()
    dispositions = df_visits_existing['ed_disposition'].value_counts(normalize=True).to_dict()
    
    rows = []
    visit_id = 900000  # Commencer après les IDs existants
    
    for _, day_context in df_context_2026.iterrows():
        date = pd.to_datetime(day_context['date'])
        day_of_week = date.dayofweek
        
        # Nombre de visites (plus élevé en hiver, surtout lundi)
        base_visits = 290  # Moyenne hivernale
        if day_of_week == 0:  # Lundi
            base_visits = 320
        elif day_of_week in [5, 6]:  # Weekend
            base_visits = 270
        
        # Ajuster selon épidémie
        epidemic_factor = 1 + (day_context['flu_level'] + day_context['bronchiolitis_level']) * 0.02
        n_visits = int(np.random.normal(base_visits * epidemic_factor, 20))
        n_visits = max(200, min(400, n_visits))
        
        # Générer chaque visite
        for _ in range(n_visits):
            visit_id += 1
            
            # Heure d'arrivée (distribution réaliste)
            hour_weights = [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.05, 
                          0.07, 0.08, 0.09, 0.09, 0.08, 0.07, 0.06, 0.05,
                          0.05, 0.05, 0.04, 0.04, 0.03, 0.02, 0.02, 0.02]
            # Normaliser pour que la somme = 1
            hour_weights = [w / sum(hour_weights) for w in hour_weights]
            hour = int(np.random.choice(range(24), p=hour_weights))
            minute = random.randint(0, 59)
            arrival_time = date + timedelta(hours=hour, minutes=minute)
            
            # Âge (distribution bimodale: enfants + personnes âgées en hiver)
            if random.random() < 0.25:  # 25% enfants (bronchiolite)
                age = np.random.choice(range(0, 10))
            elif random.random() < 0.4:  # 40% personnes âgées (grippe)
                age = int(np.random.normal(75, 12))
                age = max(60, min(100, age))
            else:
                age = int(np.random.normal(45, 18))
                age = max(18, min(90, age))
            
            sex = random.choice(['M', 'F'])
            
            # Triage (plus de cas graves en hiver)
            triage_weights = [0.03, 0.08, 0.25, 0.40, 0.24]  # 1-5
            triage_level = np.random.choice([1, 2, 3, 4, 5], p=triage_weights)
            
            # Infection suspectée (plus fréquent en hiver)
            infection_prob = 0.3 + day_context['flu_level'] * 0.05
            infection_suspected = 1 if random.random() < infection_prob else 0
            isolation_required = 1 if infection_suspected and random.random() < 0.4 else 0
            
            # Service
            service = np.random.choice(list(services.keys()), p=list(services.values()))
            
            # Mode d'arrivée
            arrival_mode = np.random.choice(list(arrival_modes.keys()), p=list(arrival_modes.values()))
            
            # Temps d'attente et séjour (plus longs quand saturé)
            occupancy_factor = day_context['beds_occupied'] / 1600
            wait_time = int(np.random.exponential(45 * occupancy_factor))
            wait_time = min(wait_time, 480)  # Max 8h
            
            los = int(np.random.exponential(180 * occupancy_factor))
            los = max(30, min(los, 1440))  # Entre 30min et 24h
            
            # Disposition
            disposition = np.random.choice(list(dispositions.keys()), p=list(dispositions.values()))
            
            # Left without being seen
            lwbs = 1 if wait_time > 240 and random.random() < 0.1 else 0
            
            rows.append({
                'visit_id': visit_id,
                'arrival_datetime': arrival_time.strftime('%Y-%m-%d %H:%M:%S'),
                'age': age,
                'sex': sex,
                'arrival_mode': arrival_mode,
                'triage_level': triage_level,
                'chief_complaint': 'respiratory' if infection_suspected else 'other',
                'infection_suspected': infection_suspected,
                'isolation_required': isolation_required,
                'service_destination': service,
                'ed_disposition': disposition,
                'los_ed_minutes': los,
                'wait_time_minutes': wait_time,
                'left_without_being_seen': lwbs
            })
    
    df_new = pd.DataFrame(rows)
    print(f"   ✓ {len(df_new)} visites générées")
    return df_new


def append_to_existing_files(df_context_new, df_visits_new):
    """Ajoute les nouvelles données aux fichiers existants."""
    print("\n📝 Mise à jour des fichiers...")
    
    # === daily_context.csv ===
    df_context_existing = pd.read_csv(RAW_PATH / "daily_context.csv")
    df_context_combined = pd.concat([df_context_existing, df_context_new], ignore_index=True)
    df_context_combined.to_csv(RAW_PATH / "daily_context.csv", index=False)
    print(f"   ✓ daily_context.csv: {len(df_context_existing)} → {len(df_context_combined)} lignes")
    
    # === ed_visits.csv ===
    df_visits_existing = pd.read_csv(RAW_PATH / "ed_visits_pitie_salpetriere_synthetic.csv")
    df_visits_combined = pd.concat([df_visits_existing, df_visits_new], ignore_index=True)
    df_visits_combined.to_csv(RAW_PATH / "ed_visits_pitie_salpetriere_synthetic.csv", index=False)
    print(f"   ✓ ed_visits.csv: {len(df_visits_existing)} → {len(df_visits_combined)} lignes")


def main():
    print("=" * 60)
    print("GÉNÉRATION DES DONNÉES 2026")
    print("Période: 1er janvier 2026 - 5 février 2026")
    print("=" * 60 + "\n")
    
    # 1. Générer contexte journalier
    df_context_2026 = generate_daily_context_2026()
    
    # 2. Générer visites
    df_visits_2026 = generate_visits_2026(df_context_2026)
    
    # 3. Ajouter aux fichiers existants
    append_to_existing_files(df_context_2026, df_visits_2026)
    
    print("\n" + "=" * 60)
    print("✅ DONNÉES 2026 GÉNÉRÉES AVEC SUCCÈS")
    print("=" * 60)
    print("\nProchaines étapes:")
    print("1. python data/processed/preprocessing.py")
    print("2. python ml/visits_train.py")
    print("3. python ml/wait_time_train.py")
    print("4. python ml/admission_train.py")
    print("5. python ml/beds_train.py")
    print("6. uvicorn back.main:app --reload")
    print("7. streamlit run front/app.py")


if __name__ == "__main__":
    main()

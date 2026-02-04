"""
Dashboard Streamlit - Gestion des Urgences
Hôpital Pitié-Salpêtrière
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# === CONFIGURATION ===
st.set_page_config(
    page_title="Urgences Pitié-Salpêtrière",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent

# === STYLES CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .alert-high {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stMetric {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# === FONCTIONS UTILITAIRES ===

def check_api_health():
    """Vérifie si l'API est disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_prediction(endpoint: str, data: dict):
    """Appelle l'API pour une prédiction."""
    try:
        response = requests.post(f"{API_URL}/predict/{endpoint}", json=data, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Erreur API: {e}")
        return None


def load_historical_data():
    """Charge les données historiques pour les graphiques."""
    try:
        context_path = PROJECT_ROOT / "data" / "raw" / "daily_context.csv"
        if context_path.exists():
            df = pd.read_csv(context_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
    except:
        pass
    return None


def get_alert_level(occupancy_rate: float) -> tuple:
    """Retourne le niveau d'alerte basé sur le taux d'occupation."""
    if occupancy_rate >= 0.95:
        return "🔴 CRITIQUE", "alert-high", "Capacité critique - Actions urgentes requises"
    elif occupancy_rate >= 0.85:
        return "🟠 ÉLEVÉ", "alert-medium", "Surveillance accrue recommandée"
    else:
        return "🟢 NORMAL", "alert-low", "Situation sous contrôle"


# === SIDEBAR ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/3/33/Logo_AP-HP.svg/200px-Logo_AP-HP.svg.png", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Menu",
        ["🏠 Tableau de bord", "📊 Prédictions", "📈 Historique", "⚙️ Paramètres"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Status API
    api_status = check_api_health()
    if api_status:
        st.success("✅ API connectée")
    else:
        st.error("❌ API non disponible")
        st.caption("Lancez: `uvicorn back.main:app`")
    
    st.divider()
    st.caption(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("v1.0.0 - Pitié-Salpêtrière")


# === PAGE: TABLEAU DE BORD ===
if page == "🏠 Tableau de bord":
    st.markdown('<h1 class="main-header">🏥 Urgences Pitié-Salpêtrière</h1>', unsafe_allow_html=True)
    
    # Contexte actuel (simulé ou depuis l'API)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📅 Contexte du jour")
        today = datetime.now()
        
        day_of_week = st.selectbox("Jour", 
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
            index=today.weekday())
        
        is_weekend = 1 if today.weekday() >= 5 else 0
        month = today.month
        is_winter = 1 if month in [11, 12, 1, 2, 3] else 0
    
    with col2:
        st.subheader("🦠 Épidémies")
        flu_level = st.slider("Grippe", 0, 5, 2)
        bronchiolitis_level = st.slider("Bronchiolite", 0, 5, 1)
        covid_level = st.slider("COVID", 0, 5, 1)
        epidemic_score = flu_level + bronchiolitis_level + covid_level
    
    with col3:
        st.subheader("🚨 Événements")
        heatwave = st.toggle("Canicule", value=False)
        strike = st.toggle("Grève", value=False)
        mass_casualty = st.toggle("Événement majeur", value=False)
    
    st.divider()
    
    # Ressources actuelles
    st.subheader("🏗️ Ressources actuelles")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        beds_occupied = st.number_input("Lits occupés", 1000, 1800, 1550)
    with col2:
        icu_beds = st.number_input("Lits réa occupés", 50, 120, 85)
    with col3:
        doctors = st.number_input("Médecins dispo", 20, 80, 50)
    with col4:
        nurses = st.number_input("Infirmiers dispo", 50, 200, 120)
    
    # Historique visites
    col1, col2, col3 = st.columns(3)
    with col1:
        visits_yesterday = st.number_input("Visites hier", 150, 400, 285)
    with col2:
        visits_avg_7d = st.number_input("Moyenne 7j", 150, 400, 280)
    with col3:
        visits_avg_30d = st.number_input("Moyenne 30j", 150, 400, 275)
    
    st.divider()
    
    # === PRÉDICTIONS ===
    if st.button("🔮 Générer les prédictions", type="primary", use_container_width=True):
        
        context = {
            "day_of_week": ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"].index(day_of_week),
            "month": month,
            "is_weekend": is_weekend,
            "is_winter": is_winter,
            "flu_level": flu_level,
            "bronchiolitis_level": bronchiolitis_level,
            "covid_level": covid_level,
            "epidemic_score": epidemic_score,
            "heatwave": int(heatwave),
            "strike": int(strike),
            "mass_casualty_event": int(mass_casualty),
            "beds_occupied": beds_occupied,
            "icu_beds_occupied": icu_beds,
            "staff_doctors_available": doctors,
            "staff_nurses_available": nurses,
            "visits_yesterday": visits_yesterday,
            "visits_avg_7days": visits_avg_7d,
            "visits_avg_30days": visits_avg_30d
        }
        
        with st.spinner("Calcul des prédictions..."):
            # Prédictions
            visits_pred = get_prediction("visits", context)
            beds_pred = get_prediction("beds", context)
        
        if visits_pred or beds_pred:
            st.success("✅ Prédictions générées!")
            
            # KPIs
            st.subheader("📊 Prédictions pour demain")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if visits_pred:
                    delta = visits_pred["visits_predicted"] - visits_yesterday
                    st.metric(
                        "👥 Visites prévues",
                        f"{visits_pred['visits_predicted']}",
                        delta=f"{delta:+d} vs hier"
                    )
            
            with col2:
                if beds_pred:
                    delta = beds_pred["beds_predicted"] - beds_occupied
                    st.metric(
                        "🛏️ Lits occupés",
                        f"{beds_pred['beds_predicted']}",
                        delta=f"{delta:+d}"
                    )
            
            with col3:
                occupancy = beds_occupied / 1800
                st.metric(
                    "📈 Taux occupation",
                    f"{occupancy*100:.1f}%",
                    delta=f"{(occupancy-0.85)*100:+.1f}% vs cible"
                )
            
            with col4:
                st.metric(
                    "🦠 Score épidémique",
                    f"{epidemic_score}/15",
                    delta="Normal" if epidemic_score < 6 else "Élevé"
                )
            
            # Alertes
            st.subheader("⚠️ Alertes")
            level, css_class, message = get_alert_level(occupancy)
            st.markdown(f'<div class="{css_class}"><strong>{level}</strong> - {message}</div>', unsafe_allow_html=True)
            
            # Recommandations
            st.subheader("💡 Recommandations")
            recommendations = []
            
            if visits_pred and visits_pred["visits_predicted"] > 300:
                recommendations.append("📌 Prévoir du personnel supplémentaire demain")
            if epidemic_score >= 6:
                recommendations.append("📌 Activer le protocole épidémie")
            if occupancy >= 0.9:
                recommendations.append("📌 Envisager le report des admissions non urgentes")
            if heatwave:
                recommendations.append("📌 Activer le plan canicule")
            if strike:
                recommendations.append("📌 Activer le plan de continuité")
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("Aucune action urgente requise")
        
        else:
            st.warning("⚠️ API non disponible. Vérifiez que le serveur est lancé.")


# === PAGE: PRÉDICTIONS ===
elif page == "📊 Prédictions":
    st.markdown('<h1 class="main-header">📊 Prédictions détaillées</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🏥 Prédiction journalière", "👤 Prédiction patient"])
    
    with tab1:
        st.subheader("Prédire l'activité de demain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Contexte temporel**")
            day = st.selectbox("Jour de la semaine", 
                ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
            month = st.selectbox("Mois", range(1, 13), format_func=lambda x: 
                ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"][x-1])
            
            st.write("**Épidémies**")
            flu = st.slider("Niveau grippe", 0, 5, 0, key="flu_pred")
            bronch = st.slider("Niveau bronchiolite", 0, 5, 0, key="bronch_pred")
            covid = st.slider("Niveau COVID", 0, 5, 0, key="covid_pred")
        
        with col2:
            st.write("**Ressources**")
            beds = st.number_input("Lits occupés actuels", 1000, 1800, 1500, key="beds_pred")
            icu = st.number_input("Lits réa occupés", 50, 120, 80, key="icu_pred")
            docs = st.number_input("Médecins", 20, 80, 50, key="docs_pred")
            nurses_nb = st.number_input("Infirmiers", 50, 200, 120, key="nurses_pred")
            
            st.write("**Historique**")
            v_yesterday = st.number_input("Visites hier", 150, 400, 280, key="vy_pred")
        
        if st.button("🔮 Prédire", key="predict_day", type="primary"):
            data = {
                "day_of_week": ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"].index(day),
                "month": month,
                "is_weekend": 1 if day in ["Samedi", "Dimanche"] else 0,
                "is_winter": 1 if month in [11, 12, 1, 2, 3] else 0,
                "flu_level": flu,
                "bronchiolitis_level": bronch,
                "covid_level": covid,
                "heatwave": 0,
                "strike": 0,
                "mass_casualty_event": 0,
                "beds_occupied": beds,
                "icu_beds_occupied": icu,
                "staff_doctors_available": docs,
                "staff_nurses_available": nurses_nb,
                "visits_yesterday": v_yesterday,
                "visits_avg_7days": v_yesterday,
                "visits_avg_30days": v_yesterday
            }
            
            visits_result = get_prediction("visits", data)
            beds_result = get_prediction("beds", data)
            
            if visits_result or beds_result:
                col1, col2 = st.columns(2)
                with col1:
                    if visits_result:
                        st.success(f"👥 **Visites prévues demain:** {visits_result['visits_predicted']}")
                with col2:
                    if beds_result:
                        st.success(f"🛏️ **Lits occupés demain:** {beds_result['beds_predicted']}")
            else:
                st.error("Erreur lors de la prédiction")
    
    with tab2:
        st.subheader("Prédire pour un patient")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Informations patient**")
            age = st.number_input("Âge", 0, 110, 50)
            triage = st.selectbox("Niveau de triage", [1, 2, 3, 4, 5], 
                format_func=lambda x: f"Niveau {x} - {'Critique' if x==1 else 'Urgent' if x==2 else 'Modéré' if x==3 else 'Mineur' if x==4 else 'Non urgent'}")
            hour = st.slider("Heure d'arrivée", 0, 23, 14)
        
        with col2:
            st.write("**Contexte**")
            beds_pat = st.number_input("Lits occupés", 1000, 1800, 1500, key="beds_pat")
            occupancy = st.slider("Taux occupation (%)", 50, 100, 85) / 100
            flu_pat = st.slider("Niveau épidémie", 0, 10, 3)
        
        if st.button("🔮 Prédire patient", key="predict_patient", type="primary"):
            data = {
                "hour": hour,
                "day_of_week": datetime.now().weekday(),
                "is_weekend": 1 if datetime.now().weekday() >= 5 else 0,
                "is_night": 1 if hour < 7 or hour > 21 else 0,
                "is_winter": 1 if datetime.now().month in [11, 12, 1, 2, 3] else 0,
                "age": age,
                "triage_level": triage,
                "is_critical": 1 if triage <= 2 else 0,
                "flu_level": flu_pat / 2,
                "bronchiolitis_level": 0,
                "covid_level": 0,
                "beds_occupied": beds_pat,
                "occupancy_rate": occupancy,
                "staff_doctors_available": 50,
                "staff_nurses_available": 120,
                "heatwave": 0,
                "strike": 0,
                "mass_casualty_event": 0,
                "wait_time_minutes": 30
            }
            
            wait_result = get_prediction("wait-time", data)
            admission_result = get_prediction("admission", data)
            
            col1, col2 = st.columns(2)
            with col1:
                if wait_result:
                    st.info(f"⏱️ **Temps d'attente estimé:** {wait_result['wait_time_minutes']} min")
            with col2:
                if admission_result:
                    emoji = "✅" if admission_result['admitted'] else "🚪"
                    result = "Admission probable" if admission_result['admitted'] else "Sortie probable"
                    st.info(f"{emoji} **{result}** ({admission_result['probability']}%)")


# === PAGE: HISTORIQUE ===
elif page == "📈 Historique":
    st.markdown('<h1 class="main-header">📈 Données historiques</h1>', unsafe_allow_html=True)
    
    df = load_historical_data()
    
    if df is not None:
        # Filtres
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Période",
                value=(df['date'].min(), df['date'].max()),
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
        
        # Filtrer
        if len(date_range) == 2:
            mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
            df_filtered = df[mask]
        else:
            df_filtered = df
        
        # Graphiques
        tab1, tab2, tab3 = st.tabs(["📊 Vue générale", "🦠 Épidémies", "🏗️ Ressources"])
        
        with tab1:
            st.subheader("Évolution de l'occupation")
            fig = px.line(df_filtered, x='date', y='beds_occupied',
                title="Lits occupés au fil du temps",
                labels={'beds_occupied': 'Lits occupés', 'date': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Moyenne lits", f"{df_filtered['beds_occupied'].mean():.0f}")
            with col2:
                st.metric("Max lits", f"{df_filtered['beds_occupied'].max():.0f}")
            with col3:
                st.metric("Min lits", f"{df_filtered['beds_occupied'].min():.0f}")
            with col4:
                st.metric("Jours analysés", len(df_filtered))
        
        with tab2:
            st.subheader("Niveaux épidémiques")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['flu_level'],
                name='Grippe', mode='lines', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['bronchiolitis_level'],
                name='Bronchiolite', mode='lines', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['covid_level'],
                name='COVID', mode='lines', line=dict(color='purple')))
            fig.update_layout(title="Évolution des épidémies", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Ressources humaines")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df_filtered, x='staff_doctors_available',
                    title="Distribution médecins disponibles",
                    labels={'staff_doctors_available': 'Médecins'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df_filtered, x='staff_nurses_available',
                    title="Distribution infirmiers disponibles",
                    labels={'staff_nurses_available': 'Infirmiers'})
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("📂 Données historiques non trouvées")
        st.info("Placez le fichier `daily_context.csv` dans `data/raw/`")


# === PAGE: PARAMÈTRES ===
elif page == "⚙️ Paramètres":
    st.markdown('<h1 class="main-header">⚙️ Paramètres</h1>', unsafe_allow_html=True)
    
    st.subheader("🔌 Configuration API")
    
    api_url = st.text_input("URL de l'API", value=API_URL)
    
    if st.button("Tester la connexion"):
        try:
            response = requests.get(f"{api_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Connexion réussie!")
                st.json(data)
            else:
                st.error(f"❌ Erreur {response.status_code}")
        except Exception as e:
            st.error(f"❌ Impossible de se connecter: {e}")
    
    st.divider()
    
    st.subheader("📊 Modèles chargés")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models_loaded", [])
            
            if models:
                for model in models:
                    st.success(f"✅ {model}")
            else:
                st.warning("Aucun modèle chargé")
    except:
        st.warning("API non disponible")
    
    st.divider()
    
    st.subheader("ℹ️ À propos")
    st.markdown("""
    **Dashboard Urgences Pitié-Salpêtrière**
    
    Ce tableau de bord permet de:
    - 📊 Visualiser les prédictions d'affluence
    - 🛏️ Anticiper l'occupation des lits
    - 👤 Estimer le temps d'attente patient
    - 📈 Analyser les données historiques
    
    ---
    
    **Stack technique:**
    - Frontend: Streamlit
    - Backend: FastAPI
    - ML: scikit-learn (Random Forest)
    - Data: pandas, numpy
    
    ---
    
    *Projet EPITECH - 2025-2026*
    """)

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
from plotly.subplots import make_subplots
from pathlib import Path

# === CONFIGURATION ===
st.set_page_config(
    page_title="Urgences Pitié-Salpêtrière",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent

# === STYLES CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #334155;
        margin: 1rem 0 0.5rem 0;
    }
    .alert-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #dc2626;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-info {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
        border-left: 5px solid #8b5cf6;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .kpi-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        text-align: center;
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

@st.cache_data(ttl=300)
def load_historical_data():
    """Charge les données historiques."""
    try:
        context_path = PROJECT_ROOT / "data" / "raw" / "daily_context.csv"
        if context_path.exists():
            df = pd.read_csv(context_path)
            df['date'] = pd.to_datetime(df['date'])
            return df
    except:
        pass
    return None


@st.cache_data(ttl=300)
def load_patient_data():
    """Charge les données patients."""
    try:
        path = PROJECT_ROOT / "data" / "processed" / "ed_data_processed.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df
    except:
        pass
    return None


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
    except:
        pass
    return None


def get_last_known_data(df_context):
    """Récupère les dernières données connues."""
    if df_context is not None and len(df_context) > 0:
        last = df_context.iloc[-1]
        return {
            "date": last['date'],
            "beds_occupied": int(last.get('beds_occupied', 1500)),
            "icu_beds_occupied": int(last.get('icu_beds_occupied', 80)),
            "staff_doctors_available": int(last.get('staff_doctors_available', 50)),
            "staff_nurses_available": int(last.get('staff_nurses_available', 120)),
            "flu_level": float(last.get('flu_level', 0)),
            "bronchiolitis_level": float(last.get('bronchiolitis_level', 0)),
            "covid_level": float(last.get('covid_level', 0)),
            "heatwave": int(last.get('heatwave', 0)),
            "strike": int(last.get('strike', 0)),
            "mass_casualty_event": int(last.get('mass_casualty_event', 0))
        }
    return None


def generate_alerts(context: dict, predictions: dict):
    """Genere les alertes basees sur le contexte et les predictions."""
    alerts = {"critical": [], "warning": [], "info": []}
    
    # Alertes occupation
    occupancy = context.get('beds_occupied', 1500) / 1800
    if occupancy >= 0.95:
        alerts["critical"].append("SATURATION IMMINENTE - Occupation > 95%")
    elif occupancy >= 0.90:
        alerts["warning"].append("Occupation elevee (>90%) - Surveiller les admissions")
    elif occupancy >= 0.85:
        alerts["info"].append("Occupation normale-haute (>85%)")
    
    # Alertes evenements
    if context.get('heatwave'):
        alerts["warning"].append("CANICULE - Plan chaleur active recommande")
    if context.get('strike'):
        alerts["critical"].append("GREVE - Plan de continuite requis")
    if context.get('mass_casualty_event'):
        alerts["critical"].append("EVENEMENT MAJEUR - Plan blanc recommande")
    
    # Alertes epidemies
    epidemic_score = context.get('flu_level', 0) + context.get('bronchiolitis_level', 0) + context.get('covid_level', 0)
    if epidemic_score >= 10:
        alerts["critical"].append(f"EPIDEMIE SEVERE - Score {epidemic_score:.0f}/15")
    elif epidemic_score >= 6:
        alerts["warning"].append(f"Epidemie moderee - Score {epidemic_score:.0f}/15")
    
    # Alertes effectifs
    doctors = context.get('staff_doctors_available', 50)
    nurses = context.get('staff_nurses_available', 120)
    if doctors < 30:
        alerts["critical"].append(f"SOUS-EFFECTIF MEDECINS - Seulement {doctors} disponibles")
    elif doctors < 40:
        alerts["warning"].append(f"Effectif medecins limite - {doctors} disponibles")
    
    if nurses < 80:
        alerts["critical"].append(f"SOUS-EFFECTIF INFIRMIERS - Seulement {nurses} disponibles")
    elif nurses < 100:
        alerts["warning"].append(f"Effectif infirmiers limite - {nurses} disponibles")
    
    # Alertes predictions
    if predictions.get('visits'):
        visits = predictions['visits'].get('visits_predicted', 280)
        if visits > 350:
            alerts["critical"].append(f"AFFLUENCE PREVUE EXCEPTIONNELLE - {visits} patients attendus")
        elif visits > 320:
            alerts["warning"].append(f"Affluence elevee prevue - {visits} patients attendus")
    
    return alerts


def generate_recommendations(context: dict, predictions: dict, alerts: dict):
    """Genere les recommandations IA."""
    recommendations = []
    
    occupancy = context.get('beds_occupied', 1500) / 1800
    visits_pred = predictions.get('visits', {}).get('visits_predicted', 280)
    beds_pred = predictions.get('beds', {}).get('beds_predicted', 1500)
    
    # Recommandations capacite
    if occupancy >= 0.95 or beds_pred > 1750:
        recommendations.append({
            "priority": "URGENT",
            "icon": "",
            "action": "Ouvrir 20 lits supplementaires",
            "detail": "Activer l'unite de debordement ou negocier transferts"
        })
        recommendations.append({
            "priority": "URGENT", 
            "icon": "",
            "action": "Declencher plan de delestage",
            "detail": "Reporter admissions non urgentes, orienter vers hopitaux partenaires"
        })
    elif occupancy >= 0.90:
        recommendations.append({
            "priority": "HAUTE",
            "icon": "",
            "action": "Preparer 10 lits supplementaires",
            "detail": "Anticiper ouverture unite tampon"
        })
    
    # Recommandations personnel
    doctors = context.get('staff_doctors_available', 50)
    nurses = context.get('staff_nurses_available', 120)
    
    if visits_pred > 320 or len(alerts.get('critical', [])) > 1:
        needed_docs = max(0, 55 - doctors)
        needed_nurses = max(0, 140 - nurses)
        if needed_docs > 0:
            recommendations.append({
                "priority": "HAUTE",
                "icon": "",
                "action": f"Renfort +{needed_docs} medecins",
                "detail": "Rappel d'astreinte ou interimaires"
            })
        if needed_nurses > 0:
            recommendations.append({
                "priority": "HAUTE",
                "icon": "",
                "action": f"Renfort +{needed_nurses} infirmiers",
                "detail": "Heures supplementaires ou pool de remplacement"
            })
    
    # Recommandations epidemies
    epidemic_score = context.get('flu_level', 0) + context.get('bronchiolitis_level', 0) + context.get('covid_level', 0)
    if epidemic_score >= 8:
        recommendations.append({
            "priority": "HAUTE",
            "icon": "",
            "action": "Activer protocole epidemie",
            "detail": "Renforcer isolement, verifier stocks EPI"
        })
    
    # Recommandations evenements
    if context.get('heatwave'):
        recommendations.append({
            "priority": "MOYENNE",
            "icon": "",
            "action": "Activer plan canicule",
            "detail": "Climatisation, hydratation, surveillance personnes agees"
        })
    
    if not recommendations:
        recommendations.append({
            "priority": "INFO",
            "icon": "",
            "action": "Situation nominale",
            "detail": "Aucune action urgente requise - maintenir la vigilance"
        })
    
    return recommendations


# === SIDEBAR ===
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/3/33/Logo_AP-HP.svg/200px-Logo_AP-HP.svg.png", width=120)
    st.title("Urgences")
    
    page = st.radio(
        "Navigation",
        ["Dashboard", "Tendances", "Predictions"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Status API
    api_ok = check_api_health()
    if api_ok:
        st.success("API connectee")
    else:
        st.error("API hors ligne")
        st.caption("```uvicorn back.main:app```")
    
    st.divider()
    st.caption(f"{datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.caption("Pitié-Salpêtrière v2.0")


# === CHARGEMENT DES DONNÉES ===
df_context = load_historical_data()
df_patients = load_patient_data()
last_data = get_last_known_data(df_context)


# ============================================================
# PAGE 1 : DASHBOARD (PILOTAGE)
# ============================================================
if page == "Dashboard":
    st.markdown('<h1 class="main-header">Tableau de Pilotage</h1>', unsafe_allow_html=True)
    
    # === SÉLECTEUR HORIZON ===
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        horizon_type = st.radio(
            "Horizon de prediction",
            ["Demain", "Apres-demain", "Choisir une date"],
            horizontal=True,
            index=0
        )
    
    with col2:
        if horizon_type == "Choisir une date":
            selected_date = st.date_input(
                "Selectionner une date",
                value=datetime.now().date() + timedelta(days=1),
                min_value=datetime.now().date(),
                max_value=datetime.now().date() + timedelta(days=365)
            )
            target_date = datetime.combine(selected_date, datetime.min.time())
        else:
            if horizon_type == "Demain":
                target_date = datetime.now() + timedelta(days=1)
            else:  # Apres-demain
                target_date = datetime.now() + timedelta(days=2)
    
    with col3:
        st.metric("Date cible", target_date.strftime("%d/%m/%Y"))
    
    st.divider()
    
    # === PREPARATION CONTEXTE ===
    today = datetime.now()
    
    # Contexte basé sur les dernières données + ajustements
    if last_data:
        context = {
            "day_of_week": target_date.weekday(),
            "month": target_date.month,
            "is_weekend": 1 if target_date.weekday() >= 5 else 0,
            "is_winter": 1 if target_date.month in [11, 12, 1, 2, 3] else 0,
            "flu_level": last_data['flu_level'],
            "bronchiolitis_level": last_data['bronchiolitis_level'],
            "covid_level": last_data['covid_level'],
            "heatwave": last_data['heatwave'],
            "strike": last_data['strike'],
            "mass_casualty_event": last_data['mass_casualty_event'],
            "beds_occupied": last_data['beds_occupied'],
            "icu_beds_occupied": last_data['icu_beds_occupied'],
            "staff_doctors_available": last_data['staff_doctors_available'],
            "staff_nurses_available": last_data['staff_nurses_available'],
            "visits_yesterday": 280,
            "visits_avg_7days": 280,
            "visits_avg_30days": 275
        }
    else:
        context = {
            "day_of_week": target_date.weekday(),
            "month": target_date.month,
            "is_weekend": 1 if target_date.weekday() >= 5 else 0,
            "is_winter": 1 if target_date.month in [11, 12, 1, 2, 3] else 0,
            "flu_level": 1, "bronchiolitis_level": 1, "covid_level": 1,
            "heatwave": 0, "strike": 0, "mass_casualty_event": 0,
            "beds_occupied": 1500, "icu_beds_occupied": 80,
            "staff_doctors_available": 50, "staff_nurses_available": 120,
            "visits_yesterday": 280, "visits_avg_7days": 280, "visits_avg_30days": 275
        }
    
    # === PRÉDICTIONS ===
    predictions = {}
    if api_ok:
        predictions['visits'] = get_prediction("visits", context)
        predictions['beds'] = get_prediction("beds", context)
    
    # === KPIs ===
    st.markdown('<p class="sub-header">Indicateurs cles</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        visits = predictions.get('visits', {}).get('visits_predicted', 280) if predictions else 280
        st.metric("Patients prevus", visits, delta=f"{visits-280:+d}" if visits != 280 else None)
    
    with col2:
        beds = predictions.get('beds', {}).get('beds_predicted', context['beds_occupied']) if predictions else context['beds_occupied']
        st.metric("Lits occupes", beds, delta=f"{beds-context['beds_occupied']:+d}" if beds != context['beds_occupied'] else None)
    
    with col3:
        occupancy = (beds / 1800) * 100
        color = "normal" if occupancy < 85 else "off" if occupancy < 95 else "inverse"
        st.metric("Occupation", f"{occupancy:.1f}%", delta=f"{occupancy-85:+.1f}% vs cible", delta_color=color)
    
    with col4:
        st.metric("Medecins", context['staff_doctors_available'])
    
    with col5:
        st.metric("Infirmiers", context['staff_nurses_available'])
    
    st.divider()
    
    # === ALERTES ===
    alerts = generate_alerts(context, predictions)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Alertes</p>', unsafe_allow_html=True)
        
        if alerts["critical"]:
            for alert in alerts["critical"]:
                st.markdown(f'<div class="alert-critical"><strong>CRITIQUE</strong><br>{alert}</div>', unsafe_allow_html=True)
        
        if alerts["warning"]:
            for alert in alerts["warning"]:
                st.markdown(f'<div class="alert-warning"><strong>ATTENTION</strong><br>{alert}</div>', unsafe_allow_html=True)
        
        if alerts["info"]:
            for alert in alerts["info"]:
                st.markdown(f'<div class="alert-info"><strong>INFO</strong><br>{alert}</div>', unsafe_allow_html=True)
        
        if not any(alerts.values()):
            st.markdown('<div class="alert-success"><strong>RAS</strong><br>Aucune alerte en cours</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="sub-header">Recommandations IA</p>', unsafe_allow_html=True)
        
        recommendations = generate_recommendations(context, predictions, alerts)
        
        for rec in recommendations:
            priority_color = {"URGENT": "[!]", "HAUTE": "[H]", "MOYENNE": "[M]", "INFO": "[i]"}.get(rec["priority"], "[-]")
            st.markdown(f'''
                <div class="recommendation-card">
                    <strong>{priority_color} {rec["priority"]}</strong><br>
                    <b>{rec["action"]}</b><br>
                    <small>{rec["detail"]}</small>
                </div>
            ''', unsafe_allow_html=True)
    
    st.divider()
    
    # === CONTEXTE EVENEMENTS ===
    st.markdown('<p class="sub-header">Evenements et Contexte</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        epidemic_score = context['flu_level'] + context['bronchiolitis_level'] + context['covid_level']
        st.metric("Score epidemique", f"{epidemic_score:.1f}/15")
    with col2:
        st.metric("Canicule", "Oui" if context['heatwave'] else "Non")
    with col3:
        st.metric("Greve", "Oui" if context['strike'] else "Non")
    with col4:
        st.metric("Evenement majeur", "Oui" if context['mass_casualty_event'] else "Non")


# ============================================================
# PAGE 2 : ANALYSE DES TENDANCES
# ============================================================
elif page == "Tendances":
    st.markdown('<h1 class="main-header">Analyse des Tendances</h1>', unsafe_allow_html=True)
    
    if df_context is None:
        st.warning("Donnees historiques non disponibles")
        st.info("Placez `daily_context.csv` dans `data/raw/`")
        st.stop()
    
    # Préparation données
    df = df_context.copy()
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    df['season'] = df['month'].map({12: 'Hiver', 1: 'Hiver', 2: 'Hiver', 
                                     3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
                                     6: 'Été', 7: 'Été', 8: 'Été',
                                     9: 'Automne', 10: 'Automne', 11: 'Automne'})
    df['day_name'] = df['day_of_week'].map({0: 'Lun', 1: 'Mar', 2: 'Mer', 3: 'Jeu', 4: 'Ven', 5: 'Sam', 6: 'Dim'})
    df['month_name'] = df['month'].map({1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin',
                                         7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'})
    
    tab1, tab2, tab3 = st.tabs(["Saisonnalite", "Heatmaps", "Impact evenements"])
    
    # === TAB 1: SAISONNALITE ===
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ete vs Hiver")
            season_data = df.groupby('season')['beds_occupied'].mean().reindex(['Hiver', 'Printemps', 'Été', 'Automne'])
            fig = px.bar(x=season_data.index, y=season_data.values,
                        color=season_data.values, color_continuous_scale='RdYlBu_r',
                        labels={'x': 'Saison', 'y': 'Lits occupés (moyenne)'})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            delta = season_data['Hiver'] - season_data['Été']
            st.info(f"**Ecart Hiver/Ete:** +{delta:.0f} lits en hiver (+{delta/season_data['Été']*100:.1f}%)")
        
        with col2:
            st.subheader("Semaine vs Week-end")
            df['type_jour'] = df['day_of_week'].apply(lambda x: 'Week-end' if x >= 5 else 'Semaine')
            weekday_data = df.groupby('type_jour')['beds_occupied'].mean()
            fig = px.pie(values=weekday_data.values, names=weekday_data.index,
                        color_discrete_sequence=['#3b82f6', '#f59e0b'])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            delta = weekday_data['Semaine'] - weekday_data['Week-end']
            st.info(f"**Ecart:** +{delta:.0f} lits en semaine vs week-end")
        
        # Evolution mensuelle
        st.subheader("Evolution mensuelle")
        monthly = df.groupby(['year', 'month']).agg({
            'beds_occupied': 'mean',
            'flu_level': 'mean',
            'covid_level': 'mean'
        }).reset_index()
        monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['beds_occupied'],
                                  name='Lits occupés', line=dict(color='#3b82f6', width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['flu_level'],
                                  name='Grippe', line=dict(color='#ef4444', dash='dash')), secondary_y=True)
        fig.add_trace(go.Scatter(x=monthly['date'], y=monthly['covid_level'],
                                  name='COVID', line=dict(color='#8b5cf6', dash='dot')), secondary_y=True)
        fig.update_layout(height=400, title="Occupation vs Épidémies")
        fig.update_yaxes(title_text="Lits occupés", secondary_y=False)
        fig.update_yaxes(title_text="Niveau épidémie", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 2: HEATMAPS ===
    with tab2:
        st.subheader("Occupation par Mois et Jour")
        
        heatmap_data = df.pivot_table(values='beds_occupied', index='day_of_week', 
                                       columns='month', aggfunc='mean')
        heatmap_data.index = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        heatmap_data.columns = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        
        fig = px.imshow(heatmap_data, color_continuous_scale='RdYlGn_r',
                       labels=dict(x="Mois", y="Jour", color="Lits"),
                       aspect="auto")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Lecture:** 
        - Rouge = Forte occupation (pics a anticiper)
        - Vert = Faible occupation
        - Les lundis d'hiver sont souvent les plus charges
        """)
        
        # Heatmap horaire si donnees dispo
        if df_patients is not None and 'hour' in df_patients.columns:
            st.subheader("Occupation par Heure et Jour")
            hourly_data = df_patients.pivot_table(values='wait_time_minutes', 
                                                   index='hour', columns='day_of_week', aggfunc='mean')
            hourly_data.columns = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
            
            fig = px.imshow(hourly_data, color_continuous_scale='YlOrRd',
                           labels=dict(x="Jour", y="Heure", color="Attente (min)"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 3: IMPACT EVENEMENTS ===
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Impact Canicule")
            if 'heatwave' in df.columns:
                heatwave_impact = df.groupby('heatwave')['beds_occupied'].mean()
                fig = px.bar(x=['Normal', 'Canicule'], y=heatwave_impact.values,
                            color=['Normal', 'Canicule'], color_discrete_sequence=['#3b82f6', '#ef4444'])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if len(heatwave_impact) > 1:
                    delta = heatwave_impact[1] - heatwave_impact[0]
                    st.metric("Impact canicule", f"+{delta:.0f} lits", delta=f"+{delta/heatwave_impact[0]*100:.1f}%")
        
        with col2:
            st.subheader("Impact Greve")
            if 'strike' in df.columns:
                strike_impact = df.groupby('strike')['beds_occupied'].mean()
                fig = px.bar(x=['Normal', 'Grève'], y=strike_impact.values,
                            color=['Normal', 'Grève'], color_discrete_sequence=['#3b82f6', '#f59e0b'])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                if len(strike_impact) > 1:
                    delta = strike_impact[1] - strike_impact[0]
                    st.metric("Impact grève", f"+{delta:.0f} lits", delta=f"+{delta/strike_impact[0]*100:.1f}%")
        
        st.subheader("Correlation Epidemies - Occupation")
        fig = px.scatter(df, x='flu_level', y='beds_occupied', trendline='ols',
                        color='season', title="Grippe vs Occupation",
                        labels={'flu_level': 'Niveau grippe', 'beds_occupied': 'Lits occupés'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3 : PRÉDICTIONS DÉTAILLÉES
# ============================================================
elif page == "Predictions":
    st.markdown('<h1 class="main-header">Predictions Detaillees</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Courbes de prevision", "Simulateur de scenarios"])
    
    # === TAB 1: COURBES DE PREVISION ===
    with tab1:
        st.subheader("Previsions multi-horizons")
        
        if not api_ok:
            st.warning("API non connectee - Predictions simulees")
        
        # Générer prédictions pour plusieurs horizons
        horizons = [1, 2, 3, 7, 14, 30]
        predictions_list = []
        
        for h in horizons:
            target_date = datetime.now() + timedelta(days=h)
            ctx = {
                "day_of_week": target_date.weekday(),
                "month": target_date.month,
                "is_weekend": 1 if target_date.weekday() >= 5 else 0,
                "is_winter": 1 if target_date.month in [11, 12, 1, 2, 3] else 0,
                "flu_level": last_data['flu_level'] if last_data else 1,
                "bronchiolitis_level": last_data['bronchiolitis_level'] if last_data else 1,
                "covid_level": last_data['covid_level'] if last_data else 1,
                "heatwave": 0, "strike": 0, "mass_casualty_event": 0,
                "beds_occupied": last_data['beds_occupied'] if last_data else 1500,
                "icu_beds_occupied": last_data['icu_beds_occupied'] if last_data else 80,
                "staff_doctors_available": last_data['staff_doctors_available'] if last_data else 50,
                "staff_nurses_available": last_data['staff_nurses_available'] if last_data else 120,
                "visits_yesterday": 280, "visits_avg_7days": 280, "visits_avg_30days": 275
            }
            
            if api_ok:
                v = get_prediction("visits", ctx)
                b = get_prediction("beds", ctx)
                visits = v.get('visits_predicted', 280) if v else 280
                beds = b.get('beds_predicted', 1500) if b else 1500
            else:
                # Simulation
                visits = 280 + np.random.randint(-20, 30) + (10 if ctx['is_winter'] else -5)
                beds = 1500 + np.random.randint(-50, 80) + (30 if ctx['is_winter'] else -20)
            
            predictions_list.append({
                "horizon": f"J+{h}",
                "date": target_date.strftime("%d/%m"),
                "visits": visits,
                "beds": beds,
                "occupancy": beds / 1800 * 100
            })
        
        df_pred = pd.DataFrame(predictions_list)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_pred['horizon'], y=df_pred['visits'],
                                      mode='lines+markers', name='Patients',
                                      line=dict(color='#3b82f6', width=3),
                                      marker=dict(size=10)))
            # Intervalle de confiance simulé
            fig.add_trace(go.Scatter(x=df_pred['horizon'], y=df_pred['visits'] * 1.1,
                                      mode='lines', name='IC+', line=dict(dash='dash', color='#93c5fd')))
            fig.add_trace(go.Scatter(x=df_pred['horizon'], y=df_pred['visits'] * 0.9,
                                      mode='lines', name='IC-', line=dict(dash='dash', color='#93c5fd'),
                                      fill='tonexty', fillcolor='rgba(147, 197, 253, 0.3)'))
            fig.update_layout(title="Patients prevus", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_pred['horizon'], y=df_pred['occupancy'],
                                      mode='lines+markers', name='Occupation %',
                                      line=dict(color='#8b5cf6', width=3),
                                      marker=dict(size=10)))
            fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Cible 85%")
            fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Critique 95%")
            fig.update_layout(title="Taux d'occupation prevu", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau recapitulatif
        st.subheader("Tableau des previsions")
        st.dataframe(df_pred, use_container_width=True, hide_index=True)
    
    # === TAB 2: SIMULATEUR ===
    with tab2:
        st.subheader("Simulateur de scenarios")
        st.markdown("Ajustez les parametres pour voir l'impact sur les predictions.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Temporel**")
            sim_day = st.selectbox("Jour", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
            sim_month = st.selectbox("Mois", list(range(1, 13)), format_func=lambda x: ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'][x-1])
        
        with col2:
            st.markdown("**Epidemies**")
            sim_flu = st.slider("Grippe", 0, 5, 1)
            sim_bronch = st.slider("Bronchiolite", 0, 5, 1)
            sim_covid = st.slider("COVID", 0, 5, 1)
        
        with col3:
            st.markdown("**Evenements**")
            sim_heatwave = st.toggle("Canicule", value=False)
            sim_strike = st.toggle("Grève", value=False)
            sim_mass = st.toggle("Événement majeur", value=False)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ressources actuelles**")
            sim_beds = st.number_input("Lits occupes", 1000, 1800, 1500)
            sim_docs = st.number_input("Medecins", 20, 80, 50)
            sim_nurses = st.number_input("Infirmiers", 50, 200, 120)
        
        if st.button("Lancer la simulation", type="primary", use_container_width=True):
            sim_context = {
                "day_of_week": ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"].index(sim_day),
                "month": sim_month,
                "is_weekend": 1 if sim_day in ["Samedi", "Dimanche"] else 0,
                "is_winter": 1 if sim_month in [11, 12, 1, 2, 3] else 0,
                "flu_level": sim_flu,
                "bronchiolitis_level": sim_bronch,
                "covid_level": sim_covid,
                "heatwave": int(sim_heatwave),
                "strike": int(sim_strike),
                "mass_casualty_event": int(sim_mass),
                "beds_occupied": sim_beds,
                "icu_beds_occupied": 80,
                "staff_doctors_available": sim_docs,
                "staff_nurses_available": sim_nurses,
                "visits_yesterday": 280,
                "visits_avg_7days": 280,
                "visits_avg_30days": 275
            }
            
            st.divider()
            st.subheader("Resultats de la simulation")
            
            col1, col2, col3, col4 = st.columns(4)
            
            if api_ok:
                sim_visits = get_prediction("visits", sim_context)
                sim_beds_pred = get_prediction("beds", sim_context)
                
                with col1:
                    v = sim_visits.get('visits_predicted', 280) if sim_visits else 280
                    st.metric("Patients", v, delta=f"{v-280:+d} vs normal")
                with col2:
                    b = sim_beds_pred.get('beds_predicted', 1500) if sim_beds_pred else 1500
                    st.metric("Lits", b, delta=f"{b-sim_beds:+d}")
                with col3:
                    occ = b / 1800 * 100
                    st.metric("Occupation", f"{occ:.1f}%")
                with col4:
                    staff_need = max(0, v - 280) // 10
                    st.metric("Staff supp.", f"+{staff_need}" if staff_need > 0 else "0")
            else:
                st.warning("API non disponible pour la simulation")
            
            # Alertes du scénario
            sim_alerts = generate_alerts(sim_context, {"visits": sim_visits, "beds": sim_beds_pred} if api_ok else {})
            sim_recs = generate_recommendations(sim_context, {"visits": sim_visits, "beds": sim_beds_pred} if api_ok else {}, sim_alerts)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Alertes du scenario**")
                for a in sim_alerts.get("critical", []):
                    st.error(a)
                for a in sim_alerts.get("warning", []):
                    st.warning(a)
            
            with col2:
                st.markdown("**Recommandations**")
                for r in sim_recs[:3]:
                    st.info(f"**{r['action']}** - {r['detail']}")

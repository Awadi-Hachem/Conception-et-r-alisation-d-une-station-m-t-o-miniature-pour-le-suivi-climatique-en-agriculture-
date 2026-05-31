
from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover
    def st_autorefresh(*args, **kwargs):
        return None

from agri_schema import CORE_SENSOR_COLUMNS, DEFAULT_DB_PATH, FEATURE_COLUMNS, FRIENDLY_FEATURE_NAMES
from database import get_last_record_time, list_station_ids, load_history_df
from ml_model import RiskPredictor


st.set_page_config(
    page_title="Smart Agriculture AI Control Center",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    .app-hero {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f766e 0%, #22c55e 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(15, 118, 110, 0.15);
    }
    .app-hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .app-hero p {
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    .station-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    .station-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .station-name {
        font-weight: 800;
        font-size: 1rem;
        color: #0f172a;
    }
    .station-id {
        color: #475569;
        font-size: 0.85rem;
        margin-top: 0.1rem;
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-online {
        background: #dcfce7;
        color: #166534;
    }
    .badge-offline {
        background: #fee2e2;
        color: #991b1b;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem 0.8rem;
        margin-top: 0.75rem;
    }
    .metric-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 0.55rem 0.7rem;
        border: 1px solid #e2e8f0;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 700;
    }
    .alert-card {
        background: white;
        border-left: 6px solid #ef4444;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border: 1px solid #fee2e2;
        margin-bottom: 0.7rem;
        box-shadow: 0 6px 12px rgba(239, 68, 68, 0.06);
    }
    .rec-card {
        background: white;
        border-left: 6px solid #0ea5e9;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border: 1px solid #dbeafe;
        margin-bottom: 0.7rem;
        box-shadow: 0 6px 12px rgba(14, 165, 233, 0.06);
    }
    .small-note {
        color: #475569;
        font-size: 0.88rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

FRIENDLY_COLUMNS = {
    "station_id": "Station",
    "station_name": "Nom station",
    "ts": "Horodatage",
    "temperature_c": "Température (°C)",
    "humidity_pct": "Humidité air (%)",
    "soil_moisture_pct": "Humidité sol (%)",
    "rain_detected": "Pluie",
    "rain_analog": "Intensité pluie",
    "water_level_pct": "Niveau eau (%)",
    "nitrogen": "Azote N",
    "phosphorus": "Phosphore P",
    "potassium": "Potassium K",
    "battery_v": "Batterie (V)",
    "signal_rssi": "RSSI (dBm)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "status": "Statut",
    "age_min": "Retard (min)",
    "drought_prediction": "Prédiction sécheresse",
    "drought_probability": "Probabilité sécheresse (%)",
    "disease_prediction": "Prédiction maladie",
    "disease_probability": "Probabilité maladie (%)",
    "completeness_pct": "Complétude IA (%)",
    "missing_features": "Champs manquants",
    "confidence_level": "Confiance IA",
    "ai_mode": "Mode IA",
    "recommendation_ai": "Recommandation IA",
}


@st.cache_data(ttl=30)
def cached_station_ids(db_path: str) -> List[str]:
    return list_station_ids(db_path)


@st.cache_data(ttl=5)
def cached_history(db_path: str, hours: int) -> pd.DataFrame:
    return load_history_df(db_path, hours)


@st.cache_resource(show_spinner=False)
def load_predictor_safe() -> RiskPredictor | None:
    try:
        return RiskPredictor.load()
    except Exception:
        return None


def format_value(value: float | int | None, unit: str = "", digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.{digits}f}{unit}"


def prepare_latest(df: pd.DataFrame, offline_after_min: int) -> pd.DataFrame:
    latest = df.sort_values("ts").groupby("station_id", as_index=False).tail(1).copy()
    latest["station_name"] = latest["station_name"].fillna(latest["station_id"])
    now_utc = pd.Timestamp.now(tz="UTC")
    latest["age_min"] = ((now_utc - latest["ts"]).dt.total_seconds() / 60).round(1)
    latest["status"] = np.where(latest["age_min"] <= offline_after_min, "ONLINE", "OFFLINE")
    latest["status_order"] = np.where(latest["status"] == "ONLINE", 0, 1)
    return latest.sort_values(["status_order", "station_id"], ascending=[True, True]).drop(columns=["status_order"]).copy()


def line_chart(df: pd.DataFrame, y_col: str, title: str, y_label: str):
    chart_df = df.dropna(subset=[y_col]).copy()
    if chart_df.empty:
        return None
    chart_df["station_display"] = chart_df["station_name"].fillna(chart_df["station_id"])
    fig = px.line(
        chart_df,
        x="ts",
        y=y_col,
        color="station_display",
        markers=True,
        title=title,
        template="plotly_white",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="Station")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=y_label)
    return fig


def probability_chart(df: pd.DataFrame, probability_col: str, title: str):
    chart_df = df.dropna(subset=[probability_col]).copy()
    if chart_df.empty:
        return None
    fig = px.line(
        chart_df,
        x="ts",
        y=probability_col,
        color="station_name",
        markers=True,
        title=title,
        template="plotly_white",
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="Station")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Probabilité (%)", range=[0, 100])
    return fig


def npk_chart(latest: pd.DataFrame):
    cols = ["nitrogen", "phosphorus", "potassium"]
    chart_df = latest[["station_name", *cols]].dropna(how="all", subset=cols).copy()
    if chart_df.empty:
        return None
    melted = chart_df.melt(id_vars=["station_name"], value_vars=cols, var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map({
        "nitrogen": "Azote N",
        "phosphorus": "Phosphore P",
        "potassium": "Potassium K",
    })
    fig = px.bar(
        melted,
        x="station_name",
        y="value",
        color="metric",
        barmode="group",
        title="Derniers niveaux NPK par station",
        template="plotly_white",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="Capteur")
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Valeur")
    return fig


def rain_events_chart(history: pd.DataFrame, hours: int):
    chart_df = history.dropna(subset=["rain_detected"]).copy()
    if chart_df.empty:
        return None
    chart_df["rain_detected"] = chart_df["rain_detected"].fillna(0).astype(int)
    summary = chart_df.groupby("station_name", as_index=False)["rain_detected"].sum().rename(columns={"rain_detected": "detections"})
    if summary.empty:
        return None
    fig = px.bar(summary, x="station_name", y="detections", title=f"Détections de pluie sur les {hours} dernières heures", template="plotly_white")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Détections")
    return fig


def battery_rssi_chart(latest: pd.DataFrame):
    chart_df = latest.dropna(subset=["battery_v", "signal_rssi"], how="all").copy()
    if chart_df.empty:
        return None
    fig = px.scatter(
        chart_df,
        x="battery_v",
        y="signal_rssi",
        text="station_name",
        size=np.maximum(chart_df["soil_moisture_pct"].fillna(20), 10),
        hover_name="station_name",
        title="Santé réseau: batterie vs RSSI",
        template="plotly_white",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
    fig.update_xaxes(title="Batterie (V)")
    fig.update_yaxes(title="RSSI (dBm)")
    return fig


def build_alerts(latest: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    alerts = []
    for row in latest.itertuples(index=False):
        station_label = getattr(row, "station_name") or getattr(row, "station_id")
        station_id = getattr(row, "station_id")
        if getattr(row, "status") == "OFFLINE":
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "Critique", "message": f"Station hors ligne depuis {getattr(row, 'age_min')} min"})
        if pd.notna(getattr(row, "temperature_c")) and getattr(row, "temperature_c") > thresholds["temp_max"]:
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "Alerte", "message": f"Température élevée: {getattr(row, 'temperature_c'):.1f} °C"})
        if pd.notna(getattr(row, "soil_moisture_pct")) and getattr(row, "soil_moisture_pct") < thresholds["soil_min"]:
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "Alerte", "message": f"Humidité du sol faible: {getattr(row, 'soil_moisture_pct'):.1f} %"})
        if pd.notna(getattr(row, "water_level_pct")) and getattr(row, "water_level_pct") < thresholds["water_min"]:
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "Alerte", "message": f"Niveau d'eau bas: {getattr(row, 'water_level_pct'):.1f} %"})
        if pd.notna(getattr(row, "drought_probability")) and getattr(row, "drought_probability") >= 70:
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "IA", "message": f"IA: risque élevé de sécheresse ({getattr(row, 'drought_probability'):.1f} %)"})
        if pd.notna(getattr(row, "disease_probability")) and getattr(row, "disease_probability") >= 70:
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "IA", "message": f"IA: risque élevé de maladie ({getattr(row, 'disease_probability'):.1f} %)"})
        if getattr(row, "missing_feature_count", 0) > 0:
            alerts.append({"station_id": station_id, "station_name": station_label, "severity": "Info", "message": f"Données partielles détectées: {getattr(row, 'missing_features')}"})
    return pd.DataFrame(alerts)


def build_recommendations(latest: pd.DataFrame, thresholds: Dict[str, float]) -> List[str]:
    recommendations: List[str] = []
    for row in latest.itertuples(index=False):
        name = getattr(row, "station_name") or getattr(row, "station_id")
        soil = getattr(row, "soil_moisture_pct")
        rain = getattr(row, "rain_detected")
        water = getattr(row, "water_level_pct")
        humidity = getattr(row, "humidity_pct")
        drought_probability = getattr(row, "drought_probability", 0)
        disease_probability = getattr(row, "disease_probability", 0)
        missing_features = getattr(row, "missing_features", "Aucune")

        if pd.notna(soil) and soil < thresholds["soil_min"] and (pd.isna(rain) or int(rain) == 0):
            recommendations.append(f"{name}: déclencher une irrigation contrôlée, le sol est sec ({soil:.1f} %).")
        if pd.notna(water) and water < thresholds["water_min"]:
            recommendations.append(f"{name}: recharger la réserve d'eau ({water:.1f} % restante).")
        if drought_probability >= 65:
            recommendations.append(f"{name}: l'IA détecte un risque important de sécheresse, prioriser l'irrigation et vérifier les buses.")
        if disease_probability >= 65:
            recommendations.append(f"{name}: l'IA détecte un risque de maladie, inspecter le feuillage et améliorer l'aération.")
        if pd.notna(humidity) and humidity > 85 and disease_probability >= 50:
            recommendations.append(f"{name}: humidité élevée et risque maladie, éviter l'excès d'arrosage sur le feuillage.")
        if missing_features != "Aucune":
            recommendations.append(f"{name}: l'application reste opérationnelle mais vérifier les capteurs manquants ({missing_features}).")

    return list(dict.fromkeys(recommendations))


def render_station_cards(latest: pd.DataFrame) -> None:
    if latest.empty:
        st.info("Aucune station détectée dans la fenêtre d'analyse.")
        return

    cols = st.columns(3)
    for idx, row in latest.reset_index(drop=True).iterrows():
        badge_class = "badge-online" if row["status"] == "ONLINE" else "badge-offline"
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="station-card">
                    <div class="station-title">
                        <div>
                            <div class="station-name">{row['station_name']}</div>
                            <div class="station-id">{row['station_id']}</div>
                        </div>
                        <span class="badge {badge_class}">{row['status']}</span>
                    </div>
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-label">Température</div>
                            <div class="metric-value">{format_value(row.get('temperature_c'), ' °C')}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Humidité sol</div>
                            <div class="metric-value">{format_value(row.get('soil_moisture_pct'), ' %')}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Sécheresse IA</div>
                            <div class="metric-value">{row.get('drought_prediction', '--')} ({format_value(row.get('drought_probability'), ' %')})</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Maladie IA</div>
                            <div class="metric-value">{row.get('disease_prediction', '--')} ({format_value(row.get('disease_probability'), ' %')})</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Complétude IA</div>
                            <div class="metric-value">{format_value(row.get('completeness_pct'), ' %')}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Dernière mise à jour</div>
                            <div class="metric-value">{format_value(row.get('age_min'), ' min')}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.markdown(
    """
    <div class="app-hero">
        <h1>Smart Agriculture AI Control Center</h1>
        <p>Dashboard Streamlit multi-stations avec supervision en temps réel, IA Random Forest pour risque de sécheresse et de maladie, et fonctionnement robuste même quand certaines informations capteurs sont manquantes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Configuration")
    db_path = st.text_input("Base de données SQLite", value=os.getenv("AGRI_DB_PATH", str(DEFAULT_DB_PATH)))
    refresh_seconds = st.slider("Rafraîchissement (secondes)", min_value=2, max_value=60, value=5)
    time_window = st.select_slider("Fenêtre d'analyse", options=[6, 12, 24, 48, 72, 168], value=24, format_func=lambda x: f"{x} h" if x < 168 else "7 jours")
    offline_after_min = st.slider("Station considérée hors ligne après (min)", 2, 120, 10)

    st.subheader("Seuils métier")
    temp_max = st.slider("Température max (°C)", 20, 60, 35)
    soil_min = st.slider("Humidité sol min (%)", 5, 90, 30)
    water_min = st.slider("Niveau eau min (%)", 5, 90, 25)

    predictor = load_predictor_safe()
    if predictor is None:
        st.warning("Modèles IA non trouvés. Lance d'abord: python train_models.py")
    else:
        st.success("Modèles IA chargés")

thresholds = {
    "temp_max": float(temp_max),
    "soil_min": float(soil_min),
    "water_min": float(water_min),
}

st_autorefresh(interval=refresh_seconds * 1000, key="agri-dashboard-refresh")

all_stations = cached_station_ids(db_path)
selected_stations = st.multiselect("Filtrer les stations", options=all_stations, default=all_stations)

history = cached_history(db_path, time_window)
if selected_stations:
    history = history[history["station_id"].isin(selected_stations)].copy()

last_record_time = get_last_record_time(db_path)
if last_record_time:
    st.caption(f"Dernier paquet enregistré: {last_record_time}")

if history.empty:
    st.warning("Aucune donnée disponible pour l'instant. Lance d'abord receiver.py avec le gateway LoRa ou simulator.py pour la démo.")
    st.code(
        """
{
  "station_id": "ST-01",
  "station_name": "Parcelle Nord",
  "timestamp": "2026-04-19T10:30:00Z",
  "temperature_c": 28.4,
  "humidity_pct": 64.5,
  "soil_moisture_pct": 41.2,
  "rain_detected": 0,
  "rain_analog": 215
}
        """,
        language="json",
    )
    st.stop()

latest = prepare_latest(history, offline_after_min)
predictor = load_predictor_safe()
if predictor is not None:
    latest = predictor.predict_dataframe(latest)
    history_pred = predictor.predict_dataframe(history)
else:
    history_pred = history.copy()
    for column in ["drought_probability", "disease_probability"]:
        history_pred[column] = np.nan

alerts = build_alerts(latest, thresholds)
recommendations = build_recommendations(latest, thresholds)

station_total = int(latest["station_id"].nunique())
online_total = int((latest["status"] == "ONLINE").sum())
mean_temp = latest["temperature_c"].mean()
mean_soil = latest["soil_moisture_pct"].mean()
drought_risk_total = int((latest.get("drought_probability", pd.Series(dtype=float)) >= 50).sum()) if "drought_probability" in latest else 0
disease_risk_total = int((latest.get("disease_probability", pd.Series(dtype=float)) >= 50).sum()) if "disease_probability" in latest else 0
mean_completeness = latest.get("completeness_pct", pd.Series(dtype=float)).mean() if "completeness_pct" in latest else np.nan
active_alerts = int(alerts.shape[0])

kpi_cols = st.columns(7)
kpi_cols[0].metric("Stations online", f"{online_total}/{station_total}")
kpi_cols[1].metric("Température moyenne", format_value(mean_temp, " °C"))
kpi_cols[2].metric("Humidité sol moyenne", format_value(mean_soil, " %"))
kpi_cols[3].metric("Risque sécheresse IA", str(drought_risk_total))
kpi_cols[4].metric("Risque maladie IA", str(disease_risk_total))
kpi_cols[5].metric("Complétude IA moyenne", format_value(mean_completeness, " %"))
kpi_cols[6].metric("Alertes actives", str(active_alerts))

st.subheader("Vue instantanée des stations")
render_station_cards(latest)

overview_tab, station_tab, alert_tab, raw_tab = st.tabs(["Vue globale", "Détail station", "Alertes & recommandations", "Données brutes"])

with overview_tab:
    col1, col2 = st.columns(2)
    fig_temp = line_chart(history, "temperature_c", "Évolution de la température", "°C")
    fig_soil = line_chart(history, "soil_moisture_pct", "Évolution de l'humidité du sol", "%")
    if fig_temp is not None:
        col1.plotly_chart(fig_temp, use_container_width=True)
    else:
        col1.info("Pas de données température.")
    if fig_soil is not None:
        col2.plotly_chart(fig_soil, use_container_width=True)
    else:
        col2.info("Pas de données humidité sol.")

    col3, col4 = st.columns(2)
    fig_drought = probability_chart(history_pred, "drought_probability", "Probabilité IA de sécheresse")
    fig_disease = probability_chart(history_pred, "disease_probability", "Probabilité IA de maladie")
    if fig_drought is not None:
        col3.plotly_chart(fig_drought, use_container_width=True)
    else:
        col3.info("Pas encore de prédictions IA.")
    if fig_disease is not None:
        col4.plotly_chart(fig_disease, use_container_width=True)
    else:
        col4.info("Pas encore de prédictions IA.")

    col5, col6 = st.columns(2)
    fig_npk = npk_chart(latest)
    fig_rain = rain_events_chart(history, time_window)
    if fig_npk is not None:
        col5.plotly_chart(fig_npk, use_container_width=True)
    else:
        col5.info("Pas de données NPK disponibles.")
    if fig_rain is not None:
        col6.plotly_chart(fig_rain, use_container_width=True)
    else:
        col6.info("Pas de données pluie.")

    fig_health = battery_rssi_chart(latest)
    if fig_health is not None:
        st.plotly_chart(fig_health, use_container_width=True)

    map_df = latest.dropna(subset=["latitude", "longitude"])[["latitude", "longitude", "station_name"]].copy()
    if not map_df.empty:
        st.subheader("Carte des stations")
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))

with station_tab:
    station_choice = st.selectbox("Choisir une station", options=latest["station_id"].sort_values().tolist())
    station_history = history_pred[history_pred["station_id"] == station_choice].copy()
    station_latest = latest[latest["station_id"] == station_choice].iloc[0]

    info_cols = st.columns(4)
    info_cols[0].metric("Température", format_value(station_latest.get("temperature_c"), " °C"))
    info_cols[1].metric("Humidité air", format_value(station_latest.get("humidity_pct"), " %"))
    info_cols[2].metric("Humidité sol", format_value(station_latest.get("soil_moisture_pct"), " %"))
    info_cols[3].metric("Niveau eau", format_value(station_latest.get("water_level_pct"), " %"))

    info_cols2 = st.columns(4)
    info_cols2[0].metric("Sécheresse IA", f"{station_latest.get('drought_prediction', '--')} ({format_value(station_latest.get('drought_probability'), ' %')})")
    info_cols2[1].metric("Maladie IA", f"{station_latest.get('disease_prediction', '--')} ({format_value(station_latest.get('disease_probability'), ' %')})")
    info_cols2[2].metric("Complétude", format_value(station_latest.get("completeness_pct"), " %"))
    info_cols2[3].metric("Confiance IA", str(station_latest.get("confidence_level", "--")))

    st.markdown(
        f"<div class='small-note'><b>Statut:</b> {station_latest['status']} | <b>Retard:</b> {format_value(station_latest['age_min'], ' min')} | <b>Champs manquants:</b> {station_latest.get('missing_features', 'Aucune')}</div>",
        unsafe_allow_html=True,
    )

    dcol1, dcol2 = st.columns(2)
    fig_station_temp = line_chart(station_history, "temperature_c", f"{station_choice} - Température", "°C")
    fig_station_soil = line_chart(station_history, "soil_moisture_pct", f"{station_choice} - Humidité sol", "%")
    if fig_station_temp is not None:
        dcol1.plotly_chart(fig_station_temp, use_container_width=True)
    if fig_station_soil is not None:
        dcol2.plotly_chart(fig_station_soil, use_container_width=True)

    dcol3, dcol4 = st.columns(2)
    fig_station_drought = probability_chart(station_history, "drought_probability", f"{station_choice} - Probabilité sécheresse")
    fig_station_disease = probability_chart(station_history, "disease_probability", f"{station_choice} - Probabilité maladie")
    if fig_station_drought is not None:
        dcol3.plotly_chart(fig_station_drought, use_container_width=True)
    if fig_station_disease is not None:
        dcol4.plotly_chart(fig_station_disease, use_container_width=True)

    st.info(str(station_latest.get("recommendation_ai", "Aucune recommandation IA")))

with alert_tab:
    st.subheader("Alertes actives")
    if alerts.empty:
        st.success("Aucune alerte active avec les seuils et prédictions actuels.")
    else:
        for row in alerts.itertuples(index=False):
            st.markdown(
                f"""
                <div class="alert-card">
                    <b>{row.station_name}</b> ({row.station_id}) - <b>{row.severity}</b><br>
                    {row.message}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Recommandations automatiques")
    if recommendations:
        for item in recommendations:
            st.markdown(f"<div class='rec-card'>{item}</div>", unsafe_allow_html=True)
    else:
        st.info("Aucune recommandation particulière pour le moment.")

with raw_tab:
    st.subheader("Tableau des dernières mesures")
    display_latest = latest.copy()
    display_latest["ts"] = display_latest["ts"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    display_latest = display_latest.rename(columns=FRIENDLY_COLUMNS)
    st.dataframe(display_latest, use_container_width=True, hide_index=True)

    st.subheader("Historique détaillé")
    display_history = history_pred.copy()
    display_history["ts"] = display_history["ts"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    display_history = display_history.rename(columns=FRIENDLY_COLUMNS)
    st.dataframe(display_history, use_container_width=True, hide_index=True)

    csv_data = history_pred.copy()
    csv_data["ts"] = csv_data["ts"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    st.download_button(
        label="Télécharger l'historique CSV",
        data=csv_data.to_csv(index=False).encode("utf-8"),
        file_name="smart_agriculture_ai_history.csv",
        mime="text/csv",
    )

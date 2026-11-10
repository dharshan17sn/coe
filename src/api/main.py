"""
FastAPI Backend for Vision-Based Cyclone Monitoring & Prediction System.
North Indian Ocean (Bay of Bengal & Arabian Sea).
Serves:
- REST API for cyclone tracks, satellite feeds, predictions, and alerts.
- Live interactive Web Dashboard (Leaflet Map + Satellite Viewer + Predictions).
"""

import os
import sys
import glob
import json
import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models import TabularMLP, CustomCycloneCNN, ModifiedAlexNet
from scripts.test_vision_inference import predict_from_image

app = FastAPI(
    title="Cyclone Eye - Vision & Meteorological Prediction Platform",
    description="Real-time Cyclone Identification, Classification & Multi-Horizon Prediction for India",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRACKS_CSV = os.path.join(BASE_DIR, "datasets", "processed", "storm_tracks.csv")
CATALOG_CSV = os.path.join(BASE_DIR, "datasets", "processed", "storm_catalog.csv")
METADATA_CSV = os.path.join(BASE_DIR, "datasets", "processed", "metadata.csv")
REPORTS_CSV = os.path.join(BASE_DIR, "reports", "master_model_comparison.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "datasets", "processed", "images")
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "datasets", "raw", "insat3d")
STATIC_DIR = os.path.join(BASE_DIR, "src", "static")

# Mount Static directories
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images/processed", StaticFiles(directory=IMAGES_DIR), name="processed_images")
app.mount("/images/raw", StaticFiles(directory=RAW_IMAGES_DIR), name="raw_images")

# Load trained Tabular Model
TABULAR_MODEL_PATH = os.path.join(BASE_DIR, "models", "tabular_mlp_[physical].pt")
tabular_model = None
tabular_checkpoint = None

if os.path.exists(TABULAR_MODEL_PATH):
    try:
        tabular_checkpoint = torch.load(TABULAR_MODEL_PATH, map_location="cpu", weights_only=False)
        tabular_model = TabularMLP(in_features=len(tabular_checkpoint["features"]), hidden_dims=[128, 64, 32], dropout_rate=0.0)
        tabular_model.load_state_dict(tabular_checkpoint["model_state_dict"])
        tabular_model.eval()
        print("Tabular MLP model loaded successfully.")
    except Exception as e:
        print(f"Error loading Tabular MLP: {e}")

# In-memory alert store
ACTIVE_ALERTS = [
    {
        "alert_id": "ALT-2026-NIO-001",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "severity": "HIGH",
        "cyclone_name": "Demo Cyclone (Historical Replay: REMAL)",
        "source": "Multimodal Prediction Engine + Official IMD Reference",
        "current_wind_kt": 60,
        "current_wind_kmh": 111.1,
        "predicted_wind_12h_kt": 75,
        "predicted_wind_12h_kmh": 138.9,
        "coastal_risk_zones": ["West Bengal Coast", "North Odisha Coast", "Sundarbans"],
        "summary": "Severe Cyclonic Storm forecasted to maintain intensification before coastal crossing. Model prediction: Potential intensification to Very Severe Cyclonic Storm."
    }
]

IMD_CLASSES = [
    "Low Pressure Area",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Cyclonic Storm",
    "Super Cyclonic Storm",
]


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/api/system/status")
def get_system_status():
    """System health check and component status."""
    has_tracks = os.path.exists(TRACKS_CSV)
    models_ready = {
        "tabular_mlp_physical": os.path.exists(os.path.join(BASE_DIR, "models", "tabular_mlp_[physical].pt")),
        "tabular_mlp_no_pressure": os.path.exists(os.path.join(BASE_DIR, "models", "tabular_mlp_[no_pressure].pt")),
        "custom_cnn": os.path.exists(os.path.join(BASE_DIR, "models", "custom_cnn.pt")),
        "modified_alexnet": os.path.exists(os.path.join(BASE_DIR, "models", "modified_alexnet.pt")),
        "multimodal_fusion": os.path.exists(os.path.join(BASE_DIR, "models", "multimodal_fusion.pt")),
    }

    raw_imgs = glob.glob(os.path.join(RAW_IMAGES_DIR, "*.jpg"))
    last_capture = "N/A"
    if raw_imgs:
        last_file = max(raw_imgs, key=os.path.getmtime)
        last_capture = datetime.datetime.fromtimestamp(os.path.getmtime(last_file)).isoformat()

    return {
        "status": "ONLINE",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models_status": models_ready,
        "database": {
            "tracks_loaded": has_tracks,
            "total_cyclones": 474 if has_tracks else 0,
            "total_observations": 15654 if has_tracks else 0,
        },
        "satellite_feeds": {
            "provider": "IMD / MOSDAC (INSAT-3D / INSAT-3DR)",
            "channels": ["IR1 (10.8 um)", "VIS (Visible)", "Rapid Scan IR"],
            "last_satellite_capture": last_capture,
            "cached_raw_images": len(raw_imgs),
        },
        "mode": "Research Prototype & Real-Time Monitoring",
    }


@app.get("/api/cyclones")
def list_cyclones(
    season: Optional[int] = Query(None, description="Filter by year (e.g. 2023, 2024)"),
    basin: Optional[str] = Query(None, description="BB (Bay of Bengal) or AS (Arabian Sea)"),
    search: Optional[str] = Query(None, description="Search by cyclone name"),
    limit: int = 50
):
    """Lists cyclones from the catalog."""
    if not os.path.exists(CATALOG_CSV):
        raise HTTPException(status_code=404, detail="Storm catalog not found.")

    df = pd.read_csv(CATALOG_CSV)
    if season:
        df = df[df["SEASON"] == season]
    if basin:
        df = df[df["SUBBASIN"].str.contains(basin, case=False, na=False)]
    if search:
        df = df[df["STORM_NAME"].str.contains(search, case=False, na=False)]

    records = df.head(limit).to_dict(orient="records")
    return {"count": len(records), "cyclones": records}


@app.get("/api/cyclones/active")
def get_active_cyclone():
    """Returns the most recent monitored cyclone for the live dashboard."""
    if not os.path.exists(CATALOG_CSV):
        raise HTTPException(status_code=404, detail="Storm catalog not found.")

    df = pd.read_csv(CATALOG_CSV)
    # Get most recent named storm with highest observations
    recent = df[(df["SEASON"] >= 2023) & (df["STORM_NAME"] != "NOT_NAMED")].iloc[0]
    sid = recent["SID"]

    # Get latest track point
    tracks_df = pd.read_csv(TRACKS_CSV)
    storm_pts = tracks_df[tracks_df["SID"] == sid].sort_values(by="ISO_TIME")
    latest_pt = storm_pts.iloc[-1].to_dict()

    return {
        "status": "MONITORED_EVENT",
        "cyclone_id": sid,
        "storm_name": recent["STORM_NAME"],
        "season": int(recent["SEASON"]),
        "subbasin": recent["SUBBASIN"],
        "peak_class": recent["PEAK_IMD_CLASS"],
        "current_observation": {
            "time": latest_pt["ISO_TIME"],
            "latitude": latest_pt["LAT"],
            "longitude": latest_pt["LON"],
            "wind_speed_kt": latest_pt["WIND_KTS"],
            "wind_speed_kmh": round(latest_pt["WIND_KTS"] * 1.852, 1),
            "central_pressure_hpa": latest_pt["PRES_HPA"],
            "imd_category": latest_pt["IMD_CLASS_NAME"],
            "distance_to_land_km": latest_pt["DIST2LAND"],
        }
    }


@app.get("/api/cyclones/{sid}/track")
def get_cyclone_track(sid: str):
    """Returns the full historical track points for Leaflet map overlay."""
    if not os.path.exists(TRACKS_CSV):
        raise HTTPException(status_code=404, detail="Tracks not found.")

    df = pd.read_csv(TRACKS_CSV)
    storm_pts = df[df["SID"] == sid].sort_values(by="ISO_TIME")
    if storm_pts.empty:
        raise HTTPException(status_code=404, detail=f"Cyclone {sid} not found.")

    points = []
    for _, r in storm_pts.iterrows():
        points.append({
            "time": r["ISO_TIME"],
            "lat": r["LAT"],
            "lon": r["LON"],
            "wind_kt": r["WIND_KTS"],
            "wind_kmh": round(r["WIND_KTS"] * 1.852, 1) if pd.notna(r["WIND_KTS"]) else None,
            "pres_hpa": r["PRES_HPA"] if pd.notna(r["PRES_HPA"]) else None,
            "class_name": r["IMD_CLASS_NAME"],
            "dist_land_km": r["DIST2LAND"] if pd.notna(r["DIST2LAND"]) else None,
        })

    return {"cyclone_id": sid, "storm_name": storm_pts["STORM_NAME"].iloc[0], "points_count": len(points), "track": points}


@app.get("/api/cyclones/{sid}/predictions")
def get_cyclone_predictions(sid: str):
    """Generates multi-horizon forecasts (+6h, +12h, +24h) and cone of uncertainty."""
    if not os.path.exists(TRACKS_CSV):
        raise HTTPException(status_code=404, detail="Tracks not found.")

    df = pd.read_csv(TRACKS_CSV)
    storm_pts = df[df["SID"] == sid].sort_values(by="ISO_TIME")
    if storm_pts.empty:
        raise HTTPException(status_code=404, detail=f"Cyclone {sid} not found.")

    latest = storm_pts.iloc[-1]
    curr_lat = float(latest["LAT"])
    curr_lon = float(latest["LON"])
    curr_wind = float(latest["WIND_KTS"]) if pd.notna(latest["WIND_KTS"]) else 45.0
    curr_pres = float(latest["PRES_HPA"]) if pd.notna(latest["PRES_HPA"]) else 990.0

    # Model inference
    model_wind = curr_wind
    model_class = latest["IMD_CLASS_NAME"]
    confidence = 75.0

    if tabular_model and tabular_checkpoint:
        try:
            scaler_mean = np.array(tabular_checkpoint["scaler_mean"])
            scaler_scale = np.array(tabular_checkpoint["scaler_scale"])
            wind_mean = tabular_checkpoint["wind_mean"]
            wind_std = tabular_checkpoint["wind_std"]
            subbasin_code = 1.0 if latest["SUBBASIN"] == "BB" else 0.0

            raw_in = np.array([[curr_pres, curr_lat, curr_lon, float(latest["DIST2LAND"] or 0), subbasin_code]])
            scaled_in = (raw_in - scaler_mean) / scaler_scale
            with torch.no_grad():
                pw, pc = tabular_model(torch.tensor(scaled_in, dtype=torch.float32))
                model_wind = float(pw.item() * wind_std + wind_mean)
                probs = torch.softmax(pc, dim=1).numpy()[0]
                cls_idx = int(np.argmax(probs))
                model_class = IMD_CLASSES[cls_idx]
                confidence = round(float(probs[cls_idx]) * 100, 1)
        except Exception:
            pass

    # Compute trajectory extrapolations (+6h, +12h, +24h)
    # Track movement based on recent bearing
    d_lat, d_lon = 0.25, -0.30  # Standard NW movement in Bay of Bengal
    if len(storm_pts) >= 2:
        prev = storm_pts.iloc[-2]
        d_lat = (curr_lat - float(prev["LAT"]))
        d_lon = (curr_lon - float(prev["LON"]))

    forecasts = []
    horizons = [(6, 1.0, 0.4), (12, 2.0, 0.8), (24, 4.0, 1.5)]
    for h, step_mult, cone_rad in horizons:
        f_lat = round(curr_lat + d_lat * step_mult, 2)
        f_lon = round(curr_lon + d_lon * step_mult, 2)
        # Moderate delta
        f_wind = round(max(model_wind + (2.5 if curr_pres < 985 else -3.0) * (h / 12.0), 20.0), 1)

        # IMD class for future wind
        f_class = "Depression"
        if f_wind >= 120: f_class = "Super Cyclonic Storm"
        elif f_wind >= 90: f_class = "Extremely Severe Cyclonic Storm"
        elif f_wind >= 64: f_class = "Very Severe Cyclonic Storm"
        elif f_wind >= 48: f_class = "Severe Cyclonic Storm"
        elif f_wind >= 34: f_class = "Cyclonic Storm"
        elif f_wind >= 28: f_class = "Deep Depression"

        forecasts.append({
            "horizon_hours": h,
            "predicted_lat": f_lat,
            "predicted_lon": f_lon,
            "predicted_wind_kt": f_wind,
            "predicted_wind_kmh": round(f_wind * 1.852, 1),
            "predicted_class": f_class,
            "cone_radius_km": round(cone_rad * 111.0, 1),
        })

    trend = "Stable"
    if forecasts[1]["predicted_wind_kt"] - curr_wind > 5:
        trend = "Strengthening"
    elif forecasts[1]["predicted_wind_kt"] - curr_wind < -5:
        trend = "Weakening"

    return {
        "cyclone_id": sid,
        "storm_name": latest["STORM_NAME"],
        "current_observation": {
            "time": latest["ISO_TIME"],
            "lat": curr_lat,
            "lon": curr_lon,
            "wind_kt": curr_wind,
            "pres_hpa": curr_pres,
            "imd_class": latest["IMD_CLASS_NAME"],
        },
        "model_assessment": {
            "model_wind_kt": round(model_wind, 1),
            "model_class": model_class,
            "confidence_pct": confidence,
            "trend": trend,
        },
        "multi_horizon_forecasts": forecasts,
    }


@app.get("/api/cyclones/{sid}/satellite-images")
def get_satellite_images(sid: str):
    """Returns available satellite frames."""
    raw_imgs = glob.glob(os.path.join(RAW_IMAGES_DIR, "*.jpg"))
    proc_imgs = glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    images = []
    for r in raw_imgs:
        base = os.path.basename(r)
        images.append({
            "type": "Full Satellite Frame (Raw)",
            "channel": "IR1 (Infrared 10.8um)" if "IR1" in base else "VIS (Visible)",
            "filename": base,
            "url": f"/images/raw/{base}",
            "size_kb": round(os.path.getsize(r) / 1024, 1),
        })
    for p in proc_imgs:
        base = os.path.basename(p)
        images.append({
            "type": "Preprocessed Model Crop (224x224)",
            "channel": "Enhanced IR (Storm Center)",
            "filename": base,
            "url": f"/images/processed/{base}",
            "size_kb": round(os.path.getsize(p) / 1024, 1),
        })

    return {"cyclone_id": sid, "satellite_images_count": len(images), "images": images}


@app.get("/api/model/metrics")
def get_model_metrics():
    """Returns performance benchmarks comparison across all trained models."""
    if os.path.exists(REPORTS_CSV):
        df = pd.read_csv(REPORTS_CSV)
        return {"models_count": len(df), "metrics": df.to_dict(orient="records")}
    return {"metrics": []}


@app.get("/api/alerts")
def get_alerts():
    """Returns active coastal and intensity alerts."""
    return {"active_alerts_count": len(ACTIVE_ALERTS), "alerts": ACTIVE_ALERTS}


class PredictionRequest(BaseModel):
    pressure_hpa: float
    latitude: float
    longitude: float
    distance_to_land_km: float = 150.0
    subbasin: str = "BB"


@app.post("/api/predictions/run")
def run_custom_prediction(req: PredictionRequest):
    """Allows manual parameter input for on-the-fly model prediction."""
    if not tabular_model or not tabular_checkpoint:
        raise HTTPException(status_code=503, detail="Tabular MLP model not loaded.")

    try:
        scaler_mean = np.array(tabular_checkpoint["scaler_mean"])
        scaler_scale = np.array(tabular_checkpoint["scaler_scale"])
        wind_mean = tabular_checkpoint["wind_mean"]
        wind_std = tabular_checkpoint["wind_std"]
        subbasin_code = 1.0 if req.subbasin.upper() in ["BB", "BAY OF BENGAL"] else 0.0

        raw_in = np.array([[req.pressure_hpa, req.latitude, req.longitude, req.distance_to_land_km, subbasin_code]])
        scaled_in = (raw_in - scaler_mean) / scaler_scale

        with torch.no_grad():
            pw, pc = tabular_model(torch.tensor(scaled_in, dtype=torch.float32))
            pred_wind = float(pw.item() * wind_std + wind_mean)
            probs = torch.softmax(pc, dim=1).numpy()[0]
            cls_idx = int(np.argmax(probs))

        pred_wind = max(pred_wind, 17.0)
        return {
            "status": "SUCCESS",
            "predicted_wind_speed_kt": round(pred_wind, 1),
            "predicted_wind_speed_kmh": round(pred_wind * 1.852, 1),
            "predicted_imd_category": IMD_CLASSES[cls_idx],
            "confidence_pct": round(float(probs[cls_idx]) * 100, 1),
            "probabilities": {IMD_CLASSES[i]: round(float(probs[i]) * 100, 1) for i in range(len(IMD_CLASSES))},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def get_dashboard_html():
    """Serves the complete interactive cyclone monitoring dashboard."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Cyclone Platform Initializing...</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


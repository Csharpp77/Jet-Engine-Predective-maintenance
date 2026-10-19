"""
Jet Engine Predictive Maintenance - Streamlit Engineering Workstation Dashboard
NASA C-MAPSS FD001 | Remaining Useful Life Prediction

Refined Aerospace Engineering Analytics Workstation
- Typography: Inter (headings, body, UI) + IBM Plex Mono (IDs, cycles, metrics, telemetry)
- Color Palette: Dark navy (#060b14 / #0b1220), cyan (#00a6ff / #38bdf8), muted blue-gray (#7e92ab)
- Clean vector SVG iconography (no amateur emojis)
- Non-overlapping sidebar controls with dual threshold slider
- Scaled & raw sensor time-series with normalized multi-sensor mode
- 100% protection of all ML models, predictions, and datasets
"""

from pathlib import Path
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lightgbm import Booster

# ==============================================================================
# 1. PATH CONFIGURATION & FILE PROTECTION SAFEGUARDS
# ==============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Case-insensitive data directory check (Windows / Linux)
DATA_DIR = PROJECT_DIR / "Data"
if not DATA_DIR.is_dir():
    DATA_DIR = PROJECT_DIR / "data"

OUTPUTS_DIR = PROJECT_DIR / "outputs"
ASSETS_DIR = PROJECT_DIR / "assets"

MODEL_PATH = OUTPUTS_DIR / "lightgbm_model.txt"
PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
PROCESSED_TEST_PATH = DATA_DIR / "processed_test.csv"
RAW_TEST_PATH = DATA_DIR / "test_FD001.txt"

TURBOFAN_ICON_PATH = ASSETS_DIR / "turbofan_icon.png"
ENGINE_CUTAWAY_PATH = ASSETS_DIR / "engine_cutaway.png"

RAW_COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)
SENSOR_COLUMNS = [f"sensor{i}" for i in range(1, 22)]
TARGET_COLUMNS = {"RUL", "actual_RUL", "predicted_RUL"}
FORBIDDEN_FEATURES = TARGET_COLUMNS | {"engine_id"}

# NASA C-MAPSS FD001 Sensor Metadata
SENSOR_METADATA = [
    {"sensor": "sensor1", "name": "Total temperature at fan inlet", "units": "°R", "symbol": "T2"},
    {"sensor": "sensor2", "name": "Total temperature at LPC outlet", "units": "°R", "symbol": "T24"},
    {"sensor": "sensor3", "name": "Total temperature at HPC outlet", "units": "°R", "symbol": "T30"},
    {"sensor": "sensor4", "name": "Total temperature at LPT outlet", "units": "°R", "symbol": "T50"},
    {"sensor": "sensor5", "name": "Pressure at fan inlet", "units": "psia", "symbol": "P2"},
    {"sensor": "sensor6", "name": "Total pressure in bypass-duct", "units": "psia", "symbol": "P15"},
    {"sensor": "sensor7", "name": "Total pressure at HPC outlet", "units": "psia", "symbol": "P30"},
    {"sensor": "sensor8", "name": "Physical fan speed", "units": "rpm", "symbol": "Nf"},
    {"sensor": "sensor9", "name": "Physical core speed", "units": "rpm", "symbol": "Nc"},
    {"sensor": "sensor10", "name": "Engine pressure ratio (P50/P2)", "units": "--", "symbol": "epr"},
    {"sensor": "sensor11", "name": "Static pressure at HPC outlet", "units": "psia", "symbol": "Ps30"},
    {"sensor": "sensor12", "name": "Ratio of fuel flow to Ps30", "units": "pps/psia", "symbol": "phi"},
    {"sensor": "sensor13", "name": "Corrected fan speed", "units": "rpm", "symbol": "NRf"},
    {"sensor": "sensor14", "name": "Corrected core speed", "units": "rpm", "symbol": "NRc"},
    {"sensor": "sensor15", "name": "Bypass Ratio", "units": "--", "symbol": "BPR"},
    {"sensor": "sensor16", "name": "Burner fuel-air ratio", "units": "--", "symbol": "farB"},
    {"sensor": "sensor17", "name": "Bleed Enthalpy", "units": "--", "symbol": "htBleed"},
    {"sensor": "sensor18", "name": "Demanded fan speed", "units": "rpm", "symbol": "Nf_dmd"},
    {"sensor": "sensor19", "name": "Demanded corrected fan speed", "units": "rpm", "symbol": "PCNfR_dmd"},
    {"sensor": "sensor20", "name": "HPT coolant bleed", "units": "lbm/s", "symbol": "W31"},
    {"sensor": "sensor21", "name": "LPT coolant bleed", "units": "lbm/s", "symbol": "W32"},
]

# Clean vector SVG icons (replaces all emoji decorations)
SVG_ICONS = {
    "engine": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>""",
    "pulse": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
    "shield": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><polyline points="9 12 11 14 15 10"></polyline></svg>""",
    "target": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>""",
    "barchart": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>""",
    "fleet": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>""",
    "database": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    "chip": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>""",
    "info": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>""",
    "check": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>""",
    "crosshair": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="22" y1="12" x2="18" y2="12"></line><line x1="6" y1="12" x2="2" y2="12"></line><line x1="12" y1="6" x2="12" y2="2"></line><line x1="12" y1="22" x2="12" y2="18"></line></svg>""",
    "grid": """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>""",
}


# ==============================================================================
# 2. CACHED DATA & ASSET LOADERS
# ==============================================================================
@st.cache_resource
def load_model(model_path: str) -> Booster:
    """Loads the trained LightGBM booster model (read-only)."""
    return Booster(model_file=model_path)


@st.cache_data
def load_dashboard_data(
    predictions_path: str,
    processed_test_path: str,
    raw_test_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads prediction CSV, processed feature CSV, and raw FD001 test data."""
    predictions = pd.read_csv(predictions_path)
    processed_test = pd.read_csv(processed_test_path)
    raw_test = pd.read_csv(
        raw_test_path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMN_NAMES,
    )
    return predictions, processed_test, raw_test


def get_base64_image(image_path: Path) -> str:
    """Safely converts an image file to a base64 string for inline HTML rendering."""
    if image_path.is_file():
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    return ""


# ==============================================================================
# 3. HIGH-FIDELITY AEROSPACE WORKSTATION CSS (INTER + IBM PLEX MONO)
# ==============================================================================
def inject_custom_css() -> None:
    """Injects refined typography and dark aerospace workstation styling."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

        <style>
        /* Typography Variables & Core Standards */
        :root {
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --font-mono: 'IBM Plex Mono', 'JetBrains Mono', 'SF Mono', 'Roboto Mono', Consolas, monospace;
            --color-bg-app: #060b14;
            --color-bg-sidebar: #080e1b;
            --color-bg-card: #0b1220;
            --color-bg-card-subtle: #080e1b;
            --color-border: #162640;
            --color-border-subtle: #142238;
            --color-text-main: #f1f5f9;
            --color-text-secondary: #7e92ab;
            --color-text-muted: #53647c;
            --color-cyan: #00a6ff;
            --color-cyan-bright: #38bdf8;
            --color-status-monitoring: #22c55e;
            --color-status-warning: #f59e0b;
            --color-status-critical: #ef4444;
        }

        /* Global Application Reset */
        html, body, [class*="st-"] {
            font-family: var(--font-sans);
            color: var(--color-text-main);
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .stApp {
            background-color: var(--color-bg-app) !important;
            color: var(--color-text-main) !important;
        }

        header[data-testid="stHeader"] {
            background-color: var(--color-bg-app) !important;
            border-bottom: 1px solid var(--color-border-subtle) !important;
        }

        /* Sidebar Container */
        section[data-testid="stSidebar"] {
            background-color: var(--color-bg-sidebar) !important;
            border-right: 1px solid var(--color-border-subtle) !important;
            padding-top: 1rem !important;
        }

        section[data-testid="stSidebar"] div.stSelectbox label,
        section[data-testid="stSidebar"] div.stSlider label {
            font-family: var(--font-sans) !important;
            font-size: 0.78rem !important;
            font-weight: 500 !important;
            color: var(--color-text-secondary) !important;
            letter-spacing: 0.01em !important;
        }

        /* Sidebar Navigation Vertical Fix (prevents horizontal crushing) */
        section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            gap: 4px !important;
            width: 100% !important;
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label {
            display: flex !important;
            align-items: center !important;
            padding: 8px 12px !important;
            border-radius: 4px !important;
            background-color: #0b1424 !important;
            border: 1px solid var(--color-border) !important;
            color: var(--color-text-secondary) !important;
            font-family: var(--font-sans) !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.15s ease !important;
            margin: 0 !important;
        }

        section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:hover {
            color: #ffffff !important;
            background-color: #121f36 !important;
            border-color: #1e3a63 !important;
        }

        section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #13233c !important;
            color: var(--color-cyan) !important;
            border-color: var(--color-cyan) !important;
            border-left: 3px solid var(--color-cyan) !important;
            font-weight: 600 !important;
        }

        /* Top Workstation Header */
        .workstation-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 0.5rem 1rem 0.5rem;
            border-bottom: 1px solid var(--color-border-subtle);
            margin-bottom: 1.15rem;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .turbofan-logo {
            width: 70px;
            height: auto;
            filter: drop-shadow(0 0 6px rgba(0, 166, 255, 0.4));
        }

        .header-title-box h1 {
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.01em;
        }

        .header-title-box p {
            font-size: 0.78rem;
            color: var(--color-text-secondary);
            margin: 2px 0 0 0;
            font-weight: 400;
            letter-spacing: 0.01em;
        }

        .header-telemetry-badge {
            font-family: var(--font-mono);
            font-size: 0.76rem;
            font-weight: 500;
            color: var(--color-cyan-bright);
            background: #0b1424;
            border: 1px solid var(--color-border);
            padding: 6px 12px;
            border-radius: 4px;
            letter-spacing: 0.03em;
        }

        /* Standardized Workstation Cards */
        .ws-card {
            background-color: var(--color-bg-card);
            border: 1px solid var(--color-border);
            border-radius: 6px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
            position: relative;
        }

        .ws-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.85rem;
        }

        .ws-card-title-group {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--color-cyan);
        }

        .ws-card-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.01em;
        }

        .ws-card-subtitle {
            font-size: 0.73rem;
            color: var(--color-text-secondary);
            margin: 2px 0 0 0;
            font-weight: 400;
        }

        /* Engine Stats Grid */
        .engine-stat-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--color-border-subtle);
        }

        .engine-stat-item .stat-label {
            font-size: 0.71rem;
            font-weight: 500;
            color: var(--color-text-secondary);
            margin-bottom: 3px;
        }

        .engine-stat-item .stat-val {
            font-family: var(--font-mono);
            font-size: 1.35rem;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: -0.02em;
        }

        /* Cutaway Schematic Box */
        .engine-cutaway-box {
            position: relative;
            width: 100%;
            background: #070d18;
            border: 1px solid var(--color-border-subtle);
            border-radius: 4px;
            overflow: hidden;
            text-align: center;
            padding: 6px 4px 4px 4px;
        }

        .engine-cutaway-img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }

        /* Hero RUL Display */
        .hero-rul-container {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 100%;
        }

        .rul-hero-number-row {
            margin: 0.4rem 0 0.85rem 0;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }

        .rul-hero-val {
            font-family: var(--font-mono);
            font-size: 2.85rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
            letter-spacing: -0.03em;
        }

        .rul-hero-unit {
            font-family: var(--font-sans);
            font-size: 1rem;
            font-weight: 500;
            color: var(--color-text-secondary);
        }

        /* Status Badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 5px 14px;
            border-radius: 16px;
            font-family: var(--font-sans);
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 1.15rem;
            width: fit-content;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .status-monitoring {
            background-color: rgba(34, 197, 94, 0.12);
            color: var(--color-status-monitoring);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }
        .status-monitoring .status-dot {
            background-color: var(--color-status-monitoring);
            box-shadow: 0 0 6px var(--color-status-monitoring);
        }

        .status-warning {
            background-color: rgba(245, 158, 11, 0.12);
            color: var(--color-status-warning);
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .status-warning .status-dot {
            background-color: var(--color-status-warning);
            box-shadow: 0 0 6px var(--color-status-warning);
        }

        .status-critical {
            background-color: rgba(239, 68, 68, 0.12);
            color: var(--color-status-critical);
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .status-critical .status-dot {
            background-color: var(--color-status-critical);
            box-shadow: 0 0 6px var(--color-status-critical);
        }

        /* Linear Segmented Gauge */
        .gauge-wrapper {
            margin-top: 0.35rem;
            padding: 0 2px;
        }

        .gauge-track {
            height: 6px;
            width: 100%;
            background: #152238;
            border-radius: 3px;
            display: flex;
            position: relative;
            overflow: visible;
        }

        .gauge-seg-critical {
            width: 25%;
            height: 100%;
            background: var(--color-status-critical);
            border-radius: 3px 0 0 3px;
        }

        .gauge-seg-warning {
            width: 35%;
            height: 100%;
            background: var(--color-status-warning);
        }

        .gauge-seg-monitoring {
            width: 40%;
            height: 100%;
            background: var(--color-status-monitoring);
            border-radius: 0 3px 3px 0;
        }

        .gauge-thumb {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 13px;
            height: 13px;
            border-radius: 50%;
            background: var(--color-status-monitoring);
            border: 2px solid #ffffff;
            box-shadow: 0 0 6px rgba(34, 197, 94, 0.8);
            z-index: 2;
            transition: left 0.3s ease;
        }

        .gauge-ticks {
            display: flex;
            justify-content: space-between;
            font-size: 0.72rem;
            font-family: var(--font-mono);
            color: var(--color-text-secondary);
            margin-top: 6px;
            padding: 0 2px;
        }

        /* 5 Aligned KPI Cards */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-bottom: 0.9rem;
        }

        .kpi-card {
            background-color: var(--color-bg-card);
            border: 1px solid var(--color-border);
            border-radius: 6px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .kpi-card-header {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 0.74rem;
            font-weight: 500;
            color: var(--color-text-secondary);
            margin-bottom: 5px;
        }

        .kpi-card-header span.icon {
            color: var(--color-cyan);
            display: flex;
            align-items: center;
        }

        .kpi-card-val {
            font-family: var(--font-mono);
            font-size: 1.35rem;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: -0.02em;
        }

        .kpi-card-val.status-text-monitoring {
            color: var(--color-status-monitoring);
        }
        .kpi-card-val.status-text-warning {
            color: var(--color-status-warning);
        }
        .kpi-card-val.status-text-critical {
            color: var(--color-status-critical);
        }

        .kpi-card-sub {
            font-size: 0.7rem;
            color: var(--color-text-muted);
            margin-top: 3px;
        }

        /* Sidebar Dataset Metadata Card */
        .sidebar-dataset-card {
            background-color: #0b1424;
            border: 1px solid var(--color-border);
            border-radius: 6px;
            padding: 0.85rem;
            margin-top: 0.85rem;
        }

        .sidebar-dataset-card h4 {
            color: #ffffff;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0;
        }

        .sidebar-dataset-card p {
            color: var(--color-text-secondary);
            font-size: 0.72rem;
            margin: 3px 0 8px 0;
            line-height: 1.35;
        }

        .sidebar-stat-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-top: 1px solid var(--color-border-subtle);
            font-size: 0.74rem;
        }

        .sidebar-stat-row .label {
            color: var(--color-text-secondary);
        }

        .sidebar-stat-row .val {
            font-family: var(--font-mono);
            color: #ffffff;
            font-weight: 500;
        }

        /* Key Takeaways list */
        .takeaway-item {
            display: flex;
            gap: 10px;
            padding: 7px 0;
            border-bottom: 1px solid var(--color-border-subtle);
            font-size: 0.79rem;
            color: #cbd5e1;
            line-height: 1.35;
        }

        .takeaway-item:last-child {
            border-bottom: none;
        }

        .takeaway-icon {
            color: var(--color-cyan);
            flex-shrink: 0;
            margin-top: 2px;
            display: flex;
            align-items: center;
        }

        /* Streamlit Input & Multi-select Restyling */
        div[data-baseweb="select"] > div {
            background-color: #0b1424 !important;
            border-color: var(--color-border) !important;
            color: #ffffff !important;
            border-radius: 4px !important;
            font-family: var(--font-sans) !important;
        }

        div[data-baseweb="tag"] {
            background-color: #14243d !important;
            border: 1px solid var(--color-cyan) !important;
            color: var(--color-cyan) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.76rem !important;
        }

        /* Table Styling */
        div[data-testid="stDataFrame"] {
            font-family: var(--font-mono) !important;
            border: 1px solid var(--color-border) !important;
            border-radius: 4px !important;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 5px;
            height: 5px;
        }
        ::-webkit-scrollbar-track {
            background: var(--color-bg-app);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--color-border);
            border-radius: 2px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--color-cyan);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 4. ERROR & SCHEMA VALIDATION SAFEGUARDS
# ==============================================================================
def show_schema_error(dataset_name: str, missing_columns: set[str]) -> None:
    st.error(
        f"Schema Error: {dataset_name} is missing required columns: "
        f"{', '.join(sorted(missing_columns))}."
    )
    st.stop()


# ==============================================================================
# 5. AEROSPACE WORKSTATION PLOTTING ENGINE (MATPLOTLIB)
# ==============================================================================
def create_dark_figure(figsize: tuple[float, float], dpi: int = 150) -> tuple[plt.Figure, plt.Axes]:
    """Creates a consistent dark navy themed Matplotlib figure and axis."""
    plt.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["font.monospace"] = ["IBM Plex Mono", "JetBrains Mono", "DejaVu Sans Mono", "Consolas"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.tick_params(colors="#7e92ab", labelsize=8, width=0.8, length=3.5)
    for spine in ax.spines.values():
        spine.set_color("#162640")
        spine.set_linewidth(0.8)
    ax.grid(True, color="#142238", linestyle="-", linewidth=0.7, alpha=0.85)
    return fig, ax


def plot_sensor_trends(
    engine_history: pd.DataFrame,
    selected_sensors: list[str],
    selected_engine: int,
    normalized: bool = False,
) -> plt.Figure:
    """Renders dark-themed sensor line chart matching reference UI aesthetics."""
    fig, ax = create_dark_figure(figsize=(7.2, 3.4))

    sensor_colors = {
        "sensor2": "#00a6ff",   # Electric cyan
        "sensor3": "#f97316",   # Aerospace orange
        "sensor4": "#22c55e",   # Emerald green
        "sensor1": "#38bdf8",
        "sensor7": "#ec4899",
        "sensor8": "#eab308",
        "sensor9": "#06b6d4",
        "sensor11": "#f43f5e",
        "sensor12": "#10b981",
        "sensor13": "#8b5cf6",
        "sensor14": "#6366f1",
        "sensor15": "#14b8a6",
        "sensor17": "#fb923c",
        "sensor20": "#84cc16",
        "sensor21": "#0284c7",
    }
    fallback_palette = ["#00a6ff", "#f97316", "#22c55e", "#ec4899", "#eab308", "#a855f7", "#06b6d4"]

    for i, sensor in enumerate(selected_sensors):
        color = sensor_colors.get(sensor, fallback_palette[i % len(fallback_palette)])
        raw_vals = engine_history[sensor].values

        if normalized:
            v_min, v_max = raw_vals.min(), raw_vals.max()
            y_vals = ((raw_vals - v_min) / (v_max - v_min) * 100.0) if v_max > v_min else np.zeros_like(raw_vals) + 50.0
            label_text = f"{sensor} (norm)"
        else:
            y_vals = raw_vals
            label_text = sensor

        ax.plot(
            engine_history["cycle"],
            y_vals,
            color=color,
            marker="o",
            markersize=3.6,
            linewidth=1.3,
            label=label_text,
            alpha=0.95,
        )

    ax.set_xlabel("Cycle", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")
    ylabel = "Normalized Scale (0 – 100%)" if normalized else "Raw Sensor Reading"
    ax.set_ylabel(ylabel, color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(len(selected_sensors), 6),
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_color("#cbd5e1")
        text.set_fontfamily("monospace")

    fig.tight_layout()
    return fig


def plot_sensor_heatmap(
    engine_history: pd.DataFrame,
    selected_engine: int,
) -> plt.Figure:
    """Renders normalized 21-sensor heatmap across cycles matching reference UI."""
    fig, ax = create_dark_figure(figsize=(7.2, 3.4))

    cycle_count = len(engine_history)
    heatmap_matrix = np.zeros((21, cycle_count))

    for idx, sensor_name in enumerate(SENSOR_COLUMNS):
        readings = engine_history[sensor_name].values
        min_val = readings.min()
        max_val = readings.max()
        if max_val > min_val:
            heatmap_matrix[idx, :] = (readings - min_val) / (max_val - min_val)
        else:
            heatmap_matrix[idx, :] = 0.5

    cycles = engine_history["cycle"].values
    im = ax.imshow(
        heatmap_matrix,
        cmap="turbo",
        aspect="auto",
        interpolation="nearest",
        extent=[cycles.min(), cycles.max(), 21.5, 0.5],
    )

    ax.set_yticks(range(1, 22))
    ax.set_yticklabels([f"s{i}" for i in range(1, 22)], fontsize=6.8, color="#7e92ab", fontfamily="monospace")
    ax.set_xlabel("Cycle", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")
    ax.set_ylabel("Sensor Array", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=7))

    for label in ax.get_xticklabels():
        label.set_fontfamily("monospace")

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03)
    cbar.set_ticks([0.1, 0.9])
    cbar.set_ticklabels(["Lower\nvalue", "Higher\nvalue"])
    cbar.ax.tick_params(colors="#7e92ab", labelsize=7.2, length=0)
    for t in cbar.ax.get_yticklabels():
        t.set_fontfamily("sans-serif")
    cbar.outline.set_edgecolor("#162640")

    fig.tight_layout()
    return fig


def plot_model_prediction_scatter(predictions: pd.DataFrame) -> plt.Figure:
    """Renders Actual vs Predicted RUL scatter plot matching reference UI."""
    fig, ax = create_dark_figure(figsize=(4.8, 3.4))

    actual = predictions["actual_RUL"]
    pred = predictions["predicted_RUL"]
    max_val = max(actual.max(), pred.max()) + 15

    ax.scatter(
        actual,
        pred,
        color="#00a6ff",
        s=18,
        alpha=0.75,
        edgecolors="none",
        label="Engines",
        zorder=3,
    )
    ax.plot(
        [0, max_val],
        [0, max_val],
        color="#ef4444",
        linestyle="--",
        linewidth=1.3,
        label="Ideal Prediction (y = x)",
        zorder=2,
    )

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual RUL (cycles)", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")
    ax.set_ylabel("Predicted RUL (cycles)", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")

    legend = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=7.8,
        handlelength=1.4,
    )
    for text in legend.get_texts():
        text.set_color("#cbd5e1")
        text.set_fontfamily("sans-serif")

    fig.tight_layout()
    return fig


def plot_error_distribution(predictions: pd.DataFrame, rmse: float, mae: float, mean_err: float) -> plt.Figure:
    """Renders residual error histogram with embedded metrics matching reference UI."""
    fig, ax = create_dark_figure(figsize=(5.4, 3.4))

    residuals = predictions["predicted_RUL"] - predictions["actual_RUL"]

    ax.hist(
        residuals,
        bins=24,
        range=(-100, 100),
        color="#00a6ff",
        edgecolor="#0b1220",
        linewidth=0.7,
        alpha=0.88,
        zorder=3,
    )

    ax.set_xlim(-100, 100)
    ax.set_xlabel("Prediction Error (cycles)", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")
    ax.set_ylabel("Number of Engines", color="#7e92ab", fontsize=8.5, labelpad=5, fontfamily="sans-serif")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")

    stats_text = (
        f"RMSE\n{rmse:.2f}\n\n"
        f"MAE\n{mae:.2f}\n\n"
        f"Mean Error\n{mean_err:.2f}"
    )
    ax.text(
        1.05,
        0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=8.0,
        color="#94a3b8",
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#080e1b",
            edgecolor="#162640",
            linewidth=0.8,
        ),
        linespacing=1.35,
    )

    fig.tight_layout()
    return fig


# ==============================================================================
# 6. MAIN APPLICATION WORKSTATION CONTROLLER
# ==============================================================================
def main() -> None:
    st.set_page_config(
        page_title="Jet Engine Predictive Maintenance | Workstation",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Protect existing ML filesystem artifacts
    required_files = [
        MODEL_PATH,
        PREDICTIONS_PATH,
        PROCESSED_TEST_PATH,
        RAW_TEST_PATH,
    ]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        st.error(
            "Required dashboard artifacts not found:\n\n"
            + "\n".join(f"- `{path.relative_to(PROJECT_DIR)}`" for path in missing_files)
        )
        st.stop()

    try:
        model = load_model(str(MODEL_PATH))
        predictions, processed_test, raw_test = load_dashboard_data(
            str(PREDICTIONS_PATH),
            str(PROCESSED_TEST_PATH),
            str(RAW_TEST_PATH),
        )
    except Exception as error:
        st.error(f"Failed to load dashboard data: {error}")
        st.stop()

    # Validate dataset schemas
    prediction_columns = {"engine_id", "actual_RUL", "predicted_RUL"}
    missing_pred_cols = prediction_columns.difference(predictions.columns)
    if missing_pred_cols:
        show_schema_error("outputs/test_predictions.csv", missing_pred_cols)

    if "engine_id" not in processed_test.columns:
        show_schema_error("data/processed_test.csv", {"engine_id"})

    raw_columns = {"engine_id", "cycle", *SENSOR_COLUMNS}
    missing_raw_cols = raw_columns.difference(raw_test.columns)
    if missing_raw_cols:
        show_schema_error("data/test_FD001.txt", missing_raw_cols)

    feature_columns = model.feature_name()
    forbidden_features = FORBIDDEN_FEATURES.intersection(feature_columns)
    if forbidden_features:
        st.error(
            f"Trained model illegally contains target/engine IDs as features: "
            f"{', '.join(sorted(forbidden_features))}."
        )
        st.stop()

    missing_model_features = set(feature_columns).difference(processed_test.columns)
    if missing_model_features:
        show_schema_error("data/processed_test.csv", missing_model_features)

    # Standardize data types
    predictions = predictions.copy()
    predictions["engine_id"] = predictions["engine_id"].astype(int)
    processed_test = processed_test.copy()
    processed_test["engine_id"] = processed_test["engine_id"].astype(int)
    raw_test = raw_test.copy()
    raw_test["engine_id"] = raw_test["engine_id"].astype(int)
    raw_test["cycle"] = raw_test["cycle"].astype(int)

    # Compute official FD001 test metrics
    residuals = predictions["predicted_RUL"] - predictions["actual_RUL"]
    rmse = float((residuals.pow(2).mean()) ** 0.5)
    mae = float(residuals.abs().mean())
    mean_err = float(residuals.mean())
    engine_count = int(predictions["engine_id"].nunique())

    turbofan_b64 = get_base64_image(TURBOFAN_ICON_PATH)
    cutaway_b64 = get_base64_image(ENGINE_CUTAWAY_PATH)

    # ==========================================================================
    # 7. SIDEBAR CONTROLS & NAVIGATION (CLEAN & NON-OVERLAPPING)
    # ==========================================================================
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; color: #00a6ff;">
                {SVG_ICONS['engine']}
                <span style="font-size: 0.9rem; font-weight: 600; color: #ffffff; letter-spacing: -0.01em;">Engine Selection</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        engine_ids = sorted(predictions["engine_id"].unique())
        selected_engine = st.selectbox(
            "Engine ID",
            options=engine_ids,
            index=0,
            help="Select an evaluated engine from the NASA FD001 test fleet (100 engines)",
        )
        st.caption("Select an engine from the NASA FD001 test set (100 engines)")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Refined Status Threshold Slider (no overlapping text or number input buttons)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px; color: #38bdf8;">
                {SVG_ICONS['shield']}
                <span style="font-size: 0.85rem; font-weight: 600; color: #ffffff;">Status Thresholds</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        thresholds = st.slider(
            "Demonstration Limits (Cycles)",
            min_value=10,
            max_value=120,
            value=(25, 75),
            step=5,
            help="Define cycle boundaries: Critical (<= first limit), Warning (between limits), Monitoring (> second limit).",
        )
        critical_max, warning_max = thresholds

        st.markdown(
            f"""
            <div style="display: flex; gap: 6px; font-size: 0.72rem; margin-top: 4px; margin-bottom: 12px; font-family: var(--font-mono);">
                <span style="color: #ef4444; background: rgba(239, 68, 68, 0.12); padding: 2px 6px; border-radius: 3px; border: 1px solid rgba(239, 68, 68, 0.3);">≤ {critical_max}</span>
                <span style="color: #f59e0b; background: rgba(245, 158, 11, 0.12); padding: 2px 6px; border-radius: 3px; border: 1px solid rgba(245, 158, 11, 0.3);">{critical_max + 1}–{warning_max}</span>
                <span style="color: #22c55e; background: rgba(34, 197, 94, 0.12); padding: 2px 6px; border-radius: 3px; border: 1px solid rgba(34, 197, 94, 0.3);">&gt; {warning_max}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

        # Workstation Navigation
        st.markdown("<p style='font-size: 0.74rem; color: #7e92ab; margin-bottom: 4px; font-weight: 600; letter-spacing: 0.04em;'>NAVIGATION</p>", unsafe_allow_html=True)
        nav_selection = st.radio(
            "Navigation Menu",
            options=["Overview", "Engine Analysis", "Model Performance", "Dataset Information"],
            label_visibility="collapsed",
            key="workstation_nav",
        )

        # Sidebar Dataset Summary Card
        st.markdown(
            f"""
            <div class="sidebar-dataset-card">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px; color: #00a6ff;">
                    {SVG_ICONS['database']}
                    <h4 style="margin: 0; font-size: 0.82rem; color: #ffffff;">Dataset Reference</h4>
                </div>
                <div style="font-weight: 600; font-size: 0.79rem; color: #38bdf8; margin-bottom: 2px;">NASA C-MAPSS FD001</div>
                <p>High-bypass turbofan degradation benchmark.</p>
                <div class="sidebar-stat-row">
                    <span class="label">Test Engines</span>
                    <span class="val">100</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="label">Sensors</span>
                    <span class="val">21</span>
                </div>
                <div class="sidebar-stat-row">
                    <span class="label">Operating Settings</span>
                    <span class="val">3</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Engine Telemetry Calculation
    engine_history = (
        raw_test.loc[raw_test["engine_id"].eq(selected_engine)]
        .sort_values("cycle")
    )
    observed_cycles = int(len(engine_history))
    latest_cycle = int(engine_history["cycle"].max()) if not engine_history.empty else 0

    saved_pred_row = predictions.loc[predictions["engine_id"].eq(selected_engine)].iloc[0]
    saved_predicted_rul = float(saved_pred_row["predicted_RUL"])

    # Model inference check
    selected_features = processed_test.loc[
        processed_test["engine_id"].eq(selected_engine),
        feature_columns,
    ]
    model_prediction = float(model.predict(selected_features)[0])
    prediction_matches = abs(model_prediction - saved_predicted_rul) <= 1e-4

    # Determine status categorization
    if saved_predicted_rul <= critical_max:
        status_category = "Critical"
        status_css = "status-critical"
        status_text_css = "status-text-critical"
        status_sub = f"≤ {critical_max} cycles"
        gauge_thumb_color = "#ef4444"
    elif saved_predicted_rul <= warning_max:
        status_category = "Warning"
        status_css = "status-warning"
        status_text_css = "status-text-warning"
        status_sub = f"{critical_max + 1} – {warning_max} cycles"
        gauge_thumb_color = "#f59e0b"
    else:
        status_category = "Monitoring"
        status_css = "status-monitoring"
        status_text_css = "status-text-monitoring"
        status_sub = f"> {warning_max} cycles"
        gauge_thumb_color = "#22c55e"

    gauge_pct = min(max(int((saved_predicted_rul / 150.0) * 100), 4), 96)

    # ==========================================================================
    # 8. TOP WORKSTATION HEADER & BRANDING
    # ==========================================================================
    logo_html = (
        f'<img src="{turbofan_b64}" class="turbofan-logo" alt="Turbofan" />'
        if turbofan_b64
        else f'<div style="color: #00a6ff;">{SVG_ICONS["engine"]}</div>'
    )

    st.markdown(
        f"""
        <div class="workstation-header">
            <div class="header-brand">
                {logo_html}
                <div class="header-title-box">
                    <h1>Jet Engine Predictive Maintenance</h1>
                    <p>NASA C-MAPSS FD001 | Remaining Useful Life Prediction</p>
                </div>
            </div>
            <div class="header-telemetry-badge">
                ENGINE #{selected_engine:03d} &nbsp;|&nbsp; OBSERVED CYCLES: {observed_cycles}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================================================
    # 9. VIEW CONTROLLER ROUTING
    # ==========================================================================
    if nav_selection == "Overview":
        render_overview_view(
            selected_engine=selected_engine,
            observed_cycles=observed_cycles,
            latest_cycle=latest_cycle,
            cutaway_b64=cutaway_b64,
            saved_predicted_rul=saved_predicted_rul,
            status_category=status_category,
            status_css=status_css,
            status_text_css=status_text_css,
            status_sub=status_sub,
            gauge_pct=gauge_pct,
            gauge_thumb_color=gauge_thumb_color,
            critical_max=critical_max,
            warning_max=warning_max,
            rmse=rmse,
            mae=mae,
            mean_err=mean_err,
            engine_count=engine_count,
            engine_history=engine_history,
            predictions=predictions,
            prediction_matches=prediction_matches,
            model_prediction=model_prediction,
        )

    elif nav_selection == "Engine Analysis":
        render_engine_analysis_view(
            selected_engine=selected_engine,
            engine_history=engine_history,
            saved_predicted_rul=saved_predicted_rul,
            status_category=status_category,
            critical_max=critical_max,
            warning_max=warning_max,
        )

    elif nav_selection == "Model Performance":
        render_model_performance_view(
            predictions=predictions,
            model=model,
            rmse=rmse,
            mae=mae,
            mean_err=mean_err,
            engine_count=engine_count,
        )

    elif nav_selection == "Dataset Information":
        render_dataset_information_view()


# ==============================================================================
# 10. OVERVIEW VIEW (EXACT ALIGNMENT WITH REFERENCE DESIGN)
# ==============================================================================
def render_overview_view(
    selected_engine: int,
    observed_cycles: int,
    latest_cycle: int,
    cutaway_b64: str,
    saved_predicted_rul: float,
    status_category: str,
    status_css: str,
    status_text_css: str,
    status_sub: str,
    gauge_pct: int,
    gauge_thumb_color: str,
    critical_max: int,
    warning_max: int,
    rmse: float,
    mae: float,
    mean_err: float,
    engine_count: int,
    engine_history: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_matches: bool,
    model_prediction: float,
) -> None:
    """Renders 4-row workstation layout matching reference UI dashboard."""

    # --------------------------------------------------------------------------
    # ROW 1: ENGINE OVERVIEW (LEFT) + PREDICTED RUL HERO (RIGHT)
    # --------------------------------------------------------------------------
    col_engine, col_rul = st.columns([1.75, 1.0])

    with col_engine:
        cutaway_html = (
            f'<img src="{cutaway_b64}" class="engine-cutaway-img" alt="Turbofan Schematic" />'
            if cutaway_b64
            else '<div style="padding: 30px; color: #38bdf8; font-family: var(--font-mono);">[Turbofan Schematic Asset]</div>'
        )
        st.markdown(
            f"""
            <div class="ws-card">
                <div class="ws-card-header">
                    <div class="ws-card-title-group">
                        {SVG_ICONS['engine']}
                        <div>
                            <div class="ws-card-title">Engine {selected_engine}</div>
                            <div class="ws-card-subtitle">Test Engine (NASA FD001)</div>
                        </div>
                    </div>
                </div>
                <div class="engine-stat-row">
                    <div class="engine-stat-item">
                        <div class="stat-label">Observed Cycles</div>
                        <div class="stat-val">{observed_cycles}</div>
                    </div>
                    <div class="engine-stat-item">
                        <div class="stat-label">Latest Cycle</div>
                        <div class="stat-val">{latest_cycle}</div>
                    </div>
                    <div class="engine-stat-item">
                        <div class="stat-label">Sensors</div>
                        <div class="stat-val">21</div>
                    </div>
                    <div class="engine-stat-item">
                        <div class="stat-label">Operating Settings</div>
                        <div class="stat-val">3</div>
                    </div>
                </div>
                <div class="engine-cutaway-box">
                    {cutaway_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_rul:
        st.markdown(
            f"""
            <div class="ws-card" style="min-height: 275px;">
                <div class="hero-rul-container">
                    <div>
                        <div class="ws-card-header">
                            <div class="ws-card-title-group">
                                {SVG_ICONS['pulse']}
                                <div class="ws-card-title">Predicted Remaining Useful Life</div>
                            </div>
                        </div>
                        <div class="rul-hero-number-row">
                            <span class="rul-hero-val">{saved_predicted_rul:.1f}</span>
                            <span class="rul-hero-unit">cycles</span>
                        </div>
                        <div class="status-badge {status_css}">
                            <div class="status-dot"></div>
                            <span>{status_category}</span>
                        </div>
                    </div>
                    <div class="gauge-wrapper">
                        <div class="gauge-track">
                            <div class="gauge-seg-critical" title="Critical: 0-{critical_max} cycles"></div>
                            <div class="gauge-seg-warning" title="Warning: {critical_max + 1}-{warning_max} cycles"></div>
                            <div class="gauge-seg-monitoring" title="Monitoring: >{warning_max} cycles"></div>
                            <div class="gauge-thumb" style="left: {gauge_pct}%; background: {gauge_thumb_color};" title="Predicted RUL: {saved_predicted_rul:.1f} cycles"></div>
                        </div>
                        <div class="gauge-ticks">
                            <span>0</span>
                            <span>{critical_max}</span>
                            <span>{warning_max}</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # ROW 2: 5 KPI CARDS
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="icon">{SVG_ICONS['pulse']}</span>
                    <span>Predicted RUL</span>
                </div>
                <div class="kpi-card-val">{saved_predicted_rul:.1f} <span style="font-size: 0.8rem; font-weight: 400; color: #7e92ab;">cycles</span></div>
                <div class="kpi-card-sub">Inference output</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="icon">{SVG_ICONS['shield']}</span>
                    <span>Status</span>
                </div>
                <div class="kpi-card-val {status_text_css}">{status_category}</div>
                <div class="kpi-card-sub">{status_sub}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="icon">{SVG_ICONS['target']}</span>
                    <span>Test Set RMSE</span>
                </div>
                <div class="kpi-card-val">{rmse:.2f} <span style="font-size: 0.8rem; font-weight: 400; color: #7e92ab;">cycles</span></div>
                <div class="kpi-card-sub">Official FD001 test</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="icon">{SVG_ICONS['barchart']}</span>
                    <span>Test Set MAE</span>
                </div>
                <div class="kpi-card-val">{mae:.2f} <span style="font-size: 0.8rem; font-weight: 400; color: #7e92ab;">cycles</span></div>
                <div class="kpi-card-sub">Official FD001 test</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-card-header">
                    <span class="icon">{SVG_ICONS['fleet']}</span>
                    <span>Evaluated Engines</span>
                </div>
                <div class="kpi-card-val">{engine_count}</div>
                <div class="kpi-card-sub">Official FD001 test set</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # ROW 3: SENSOR TRENDS (LEFT) + ALL SENSORS HEATMAP (RIGHT)
    # --------------------------------------------------------------------------
    col_chart_left, col_chart_right = st.columns(2)

    with col_chart_left:
        col_hdr_l, col_hdr_r = st.columns([1.8, 1.2])
        with col_hdr_l:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 2px; color: #00a6ff;">
                    {SVG_ICONS['pulse']}
                    <span style="font-size: 0.9rem; font-weight: 600; color: #ffffff;">Sensor Trends</span>
                </div>
                <div style="font-size: 0.73rem; color: #7e92ab; margin-bottom: 6px;">Engine {selected_engine} – Selected Sensors over Observed Cycles</div>
                """,
                unsafe_allow_html=True,
            )
        with col_hdr_r:
            scale_mode = st.radio(
                "Scale Mode",
                options=["Raw Units", "Normalized (0–100%)"],
                horizontal=True,
                label_visibility="collapsed",
                key=f"overview_scale_{selected_engine}",
            )

        selected_sensors = st.multiselect(
            "Sensors",
            options=SENSOR_COLUMNS,
            default=["sensor2", "sensor3", "sensor4"],
            key=f"sensor_select_{selected_engine}",
            label_visibility="collapsed",
        )

        if not selected_sensors:
            selected_sensors = ["sensor2", "sensor3", "sensor4"]

        is_norm = scale_mode == "Normalized (0–100%)"
        fig_trends = plot_sensor_trends(engine_history, selected_sensors, selected_engine, normalized=is_norm)
        st.pyplot(fig_trends, use_container_width=True)
        plt.close(fig_trends)

    with col_chart_right:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 2px; color: #00a6ff;">
                {SVG_ICONS['grid']}
                <span style="font-size: 0.9rem; font-weight: 600; color: #ffffff;">All Sensors (Engine {selected_engine})</span>
            </div>
            <div style="font-size: 0.73rem; color: #7e92ab; margin-bottom: 6px;">Normalized min-max telemetry across observed cycles</div>
            """,
            unsafe_allow_html=True,
        )

        fig_heat = plot_sensor_heatmap(engine_history, selected_engine)
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # ROW 4: MODEL PREDICTION VS ACTUAL | ERROR DISTRIBUTION | KEY TAKEAWAYS
    # --------------------------------------------------------------------------
    col_p1, col_p2, col_p3 = st.columns([1.15, 1.25, 1.0])

    with col_p1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px; color: #00a6ff;">
                {SVG_ICONS['crosshair']}
                <span style="font-size: 0.88rem; font-weight: 600; color: #ffffff;">Model Prediction vs Actual (Test Set)</span>
            </div>
            <div style="font-size: 0.72rem; color: #7e92ab; margin-bottom: 4px;">Scatter plot of all 100 test fleet engines</div>
            """,
            unsafe_allow_html=True,
        )
        fig_scatter = plot_model_prediction_scatter(predictions)
        st.pyplot(fig_scatter, use_container_width=True)
        plt.close(fig_scatter)

    with col_p2:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px; color: #00a6ff;">
                {SVG_ICONS['barchart']}
                <span style="font-size: 0.88rem; font-weight: 600; color: #ffffff;">Error Distribution (Test Set)</span>
            </div>
            <div style="font-size: 0.72rem; color: #7e92ab; margin-bottom: 4px;">Residual distribution (predicted - actual)</div>
            """,
            unsafe_allow_html=True,
        )
        fig_err = plot_error_distribution(predictions, rmse, mae, mean_err)
        st.pyplot(fig_err, use_container_width=True)
        plt.close(fig_err)

    with col_p3:
        st.markdown(
            f"""
            <div class="ws-card" style="height: calc(100% - 10px); min-height: 255px; margin-top: 22px;">
                <div class="ws-card-header" style="margin-bottom: 0.5rem;">
                    <div class="ws-card-title-group">
                        {SVG_ICONS['chip']}
                        <div class="ws-card-title">Key Takeaways</div>
                    </div>
                </div>
                <div>
                    <div class="takeaway-item">
                        <span class="takeaway-icon">{SVG_ICONS['chip']}</span>
                        <div><strong>LightGBM Model:</strong> Gradient-boosted decision trees trained on C-MAPSS FD001 run-to-failure degradation data.</div>
                    </div>
                    <div class="takeaway-item">
                        <span class="takeaway-icon">{SVG_ICONS['crosshair']}</span>
                        <div><strong>Inference:</strong> Evaluates remaining useful life from latest cycle telemetry and rolling window statistics.</div>
                    </div>
                    <div class="takeaway-item">
                        <span class="takeaway-icon">{SVG_ICONS['check']}</span>
                        <div><strong>Benchmark Accuracy:</strong> Test set performance of <strong>18.67 RMSE</strong> and <strong>14.16 MAE</strong> across 100 engines.</div>
                    </div>
                    <div class="takeaway-item">
                        <span class="takeaway-icon">{SVG_ICONS['info']}</span>
                        <div><strong>Engineering Notice:</strong> Demonstration system using simulated NASA data. Status categories are illustrative indicators, not operational maintenance directives.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 11. ENGINE ANALYSIS DEEP-DIVE VIEW
# ==============================================================================
def render_engine_analysis_view(
    selected_engine: int,
    engine_history: pd.DataFrame,
    saved_predicted_rul: float,
    status_category: str,
    critical_max: int,
    warning_max: int,
) -> None:
    """Deep-dive engine telemetry analysis view."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.15rem;">
            <h2 style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin: 0; letter-spacing: -0.01em;">Engine #{selected_engine:03d} Telemetry & Degradation Deep-Dive</h2>
            <p style="color: #7e92ab; font-size: 0.78rem; margin-top: 2px;">High-resolution cycle-by-cycle telemetry inspection and sensor drift analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observed Cycles", f"{len(engine_history)}")
    c2.metric("Predicted RUL", f"{saved_predicted_rul:.1f} cycles")
    c3.metric("Status", status_category)
    c4.metric("Operating Settings", "3 (Sea-level simulated)")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Detailed sensor inspection (defaults to ONE sensor, with optional multi-sensor comparison)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 2px; color: #00a6ff;">
            {SVG_ICONS['pulse']}
            <span style="font-size: 0.92rem; font-weight: 600; color: #ffffff;">Detailed Sensor Telemetry Inspection</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.2, 1.3])
    with col_ctrl1:
        # Defaults to ONE sensor as requested
        inspect_sensors = st.multiselect(
            "Select Sensor(s)",
            options=SENSOR_COLUMNS,
            default=["sensor2"],
            help="Defaulted to one sensor. You can add more sensors for multi-channel comparative analysis.",
        )
        if not inspect_sensors:
            inspect_sensors = ["sensor2"]

    with col_ctrl2:
        norm_toggle = st.checkbox("Normalize Scales (0–100%)", value=(len(inspect_sensors) > 1), help="Aligns disparate engineering units (e.g., °R vs rpm) on a 0-100% min-max scale.")
        rolling_win = st.slider("Smoothing Window (Cycles)", min_value=1, max_value=20, value=5)

    with col_ctrl3:
        if len(inspect_sensors) == 1:
            meta = next((s for s in SENSOR_METADATA if s["sensor"] == inspect_sensors[0]), None)
            if meta:
                st.markdown(
                    f"""
                    <div class="ws-card" style="padding: 10px 14px; margin-bottom: 0;">
                        <div style="font-family: var(--font-mono); font-size: 0.8rem; font-weight: 600; color: #38bdf8;">{inspect_sensors[0].upper()} ({meta['symbol']})</div>
                        <div style="font-size: 0.78rem; color: #ffffff;">{meta['name']}</div>
                        <div style="font-size: 0.72rem; color: #7e92ab; margin-top: 3px;">Units: <strong>{meta['units']}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="ws-card" style="padding: 10px 14px; margin-bottom: 0;">
                    <div style="font-size: 0.78rem; font-weight: 600; color: #38bdf8;">Multi-Channel Telemetry</div>
                    <div style="font-size: 0.72rem; color: #7e92ab; margin-top: 2px;">Normalized mode active to prevent scale distortion across disparate units.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fig, ax = create_dark_figure(figsize=(9, 3.6))
    sensor_palette = ["#00a6ff", "#f97316", "#22c55e", "#ec4899", "#eab308", "#a855f7", "#06b6d4"]

    for idx, s in enumerate(inspect_sensors):
        color = sensor_palette[idx % len(sensor_palette)]
        raw_vals = engine_history[s].values
        if norm_toggle:
            v_min, v_max = raw_vals.min(), raw_vals.max()
            y_vals = ((raw_vals - v_min) / (v_max - v_min) * 100.0) if v_max > v_min else np.zeros_like(raw_vals) + 50.0
            unit_label = "%"
        else:
            y_vals = raw_vals
            meta = next((m for m in SENSOR_METADATA if m["sensor"] == s), None)
            unit_label = meta["units"] if meta else ""

        roll_vals = pd.Series(y_vals).rolling(rolling_win, min_periods=1).mean()

        if len(inspect_sensors) == 1:
            ax.plot(engine_history["cycle"], y_vals, color=color, alpha=0.4, marker=".", label=f"{s} Raw ({unit_label})")
            ax.plot(engine_history["cycle"], roll_vals, color="#f97316", linewidth=1.8, label=f"Rolling Mean (w={rolling_win})")
        else:
            ax.plot(engine_history["cycle"], roll_vals, color=color, linewidth=1.6, label=f"{s} (w={rolling_win})")

    ax.set_xlabel("Cycle", color="#7e92ab", fontsize=8.5, fontfamily="sans-serif")
    y_title = "Normalized Scale (0 – 100%)" if norm_toggle else f"Sensor Reading ({unit_label})"
    ax.set_ylabel(y_title, color="#7e92ab", fontsize=8.5, fontfamily="sans-serif")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")
    ax.legend(facecolor="#0b1220", edgecolor="#162640", labelcolor="#cbd5e1", fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Telemetry statistics summary table
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 7px; margin-bottom: 6px; color: #00a6ff;">
            {SVG_ICONS['database']}
            <span style="font-size: 0.92rem; font-weight: 600; color: #ffffff;">Observed Telemetry Summary Table</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_data = []
    for s in SENSOR_COLUMNS:
        vals = engine_history[s]
        init_val = vals.iloc[0]
        final_val = vals.iloc[-1]
        delta = final_val - init_val
        meta = next((m for m in SENSOR_METADATA if m["sensor"] == s), None)
        summary_data.append({
            "Sensor": s,
            "Symbol": meta["symbol"] if meta else "--",
            "Units": meta["units"] if meta else "--",
            "Initial": f"{init_val:.2f}",
            "Latest": f"{final_val:.2f}",
            "Drift": f"{delta:+.2f}",
            "Min": f"{vals.min():.2f}",
            "Max": f"{vals.max():.2f}",
            "Std Dev": f"{vals.std():.2f}",
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# ==============================================================================
# 12. MODEL PERFORMANCE DEEP-DIVE VIEW
# ==============================================================================
def render_model_performance_view(
    predictions: pd.DataFrame,
    model: Booster,
    rmse: float,
    mae: float,
    mean_err: float,
    engine_count: int,
) -> None:
    """Deep-dive model evaluation, error breakdown, and feature importance view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.15rem;">
            <h2 style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin: 0; letter-spacing: -0.01em;">LightGBM Model Architecture & FD001 Benchmark Evaluation</h2>
            <p style="color: #7e92ab; font-size: 0.78rem; margin-top: 2px;">Comprehensive residual diagnostics and tree split-gain feature importance analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Root Mean Squared Error", f"{rmse:.2f} cycles")
    m2.metric("Mean Absolute Error", f"{mae:.2f} cycles")
    m3.metric("Mean Residual Bias", f"{mean_err:+.2f} cycles")
    m4.metric("Test Fleet Evaluated", f"{engine_count} engines")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px; color: #00a6ff;">
                {SVG_ICONS['crosshair']}
                <span style="font-size: 0.88rem; font-weight: 600; color: #ffffff;">Actual vs Predicted Remaining Useful Life</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_scatter = plot_model_prediction_scatter(predictions)
        st.pyplot(fig_scatter, use_container_width=True)
        plt.close(fig_scatter)

    with col2:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px; color: #00a6ff;">
                {SVG_ICONS['barchart']}
                <span style="font-size: 0.88rem; font-weight: 600; color: #ffffff;">Residual Error Distribution (-100 to +100 Cycles)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_hist = plot_error_distribution(predictions, rmse, mae, mean_err)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px; color: #00a6ff;">
            {SVG_ICONS['chip']}
            <span style="font-size: 0.9rem; font-weight: 600; color: #ffffff;">Top Engineered Features by Split Gain</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        f_names = model.feature_name()
        importances = model.feature_importance(importance_type="gain")
        feat_df = pd.DataFrame({"Feature": f_names, "Gain": importances}).sort_values("Gain", ascending=False).head(12)

        fig, ax = create_dark_figure(figsize=(9, 3.6))
        y_pos = range(len(feat_df))
        ax.barh(y_pos, feat_df["Gain"][::-1], color="#00a6ff", edgecolor="#38bdf8", alpha=0.85, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_df["Feature"][::-1], fontsize=8, color="#cbd5e1", fontfamily="monospace")
        ax.set_xlabel("Split Gain Importance", color="#7e92ab", fontsize=8.5, fontfamily="sans-serif")
        for label in ax.get_xticklabels():
            label.set_fontfamily("monospace")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.info(f"Feature importance display: {e}")


# ==============================================================================
# 13. DATASET INFORMATION VIEW
# ==============================================================================
def render_dataset_information_view() -> None:
    """Comprehensive NASA C-MAPSS FD001 dataset reference view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.15rem;">
            <h2 style="color: #ffffff; font-size: 1.2rem; font-weight: 600; margin: 0; letter-spacing: -0.01em;">NASA C-MAPSS FD001 Dataset Specification</h2>
            <p style="color: #7e92ab; font-size: 0.78rem; margin-top: 2px;">Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Benchmark Documentation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ws-card">
            <h3 style="color: #38bdf8; font-size: 0.95rem; margin-top: 0; font-weight: 600;">Benchmark Overview</h3>
            <p style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.45;">
                The <strong>C-MAPSS</strong> dataset is a standard aerospace engineering benchmark provided by NASA Ames Prognostics Center of Excellence. 
                Sub-dataset <strong>FD001</strong> models a fleet of high-bypass turbofan engines operating under single flight conditions (sea-level) 
                subject to a single degradation fault mode: <strong>High-Pressure Compressor (HPC) blade degradation</strong>.
            </p>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px;">
                <div style="background: #080e1b; padding: 10px; border-radius: 4px; border: 1px solid #142238;">
                    <div style="font-size: 0.71rem; color: #7e92ab;">Operating Conditions</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">Single (Sea Level)</div>
                </div>
                <div style="background: #080e1b; padding: 10px; border-radius: 4px; border: 1px solid #142238;">
                    <div style="font-size: 0.71rem; color: #7e92ab;">Fault Modes</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">HPC Degradation</div>
                </div>
                <div style="background: #080e1b; padding: 10px; border-radius: 4px; border: 1px solid #142238;">
                    <div style="font-size: 0.71rem; color: #7e92ab;">Train Trajectories</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">100 Run-to-Failure</div>
                </div>
                <div style="background: #080e1b; padding: 10px; border-radius: 4px; border: 1px solid #142238;">
                    <div style="font-size: 0.71rem; color: #7e92ab;">Test Trajectories</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">100 Truncated</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px; color: #00a6ff;">
            {SVG_ICONS['database']}
            <span style="font-size: 0.9rem; font-weight: 600; color: #ffffff;">Turbofan Sensor Descriptions & Engineering Units</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(SENSOR_METADATA), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

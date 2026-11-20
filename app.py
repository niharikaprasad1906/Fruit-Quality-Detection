"""
Fruit Quality Grading System - Web Application
==============================================
AI-powered fruit freshness analysis using image segmentation and machine learning.
Streamlit Web Application for Windows.
"""

from pathlib import Path
import base64
import time
import io
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Application modules
from app_modules.model_loader import (
    load_trained_model,
    CLASS_NAMES,
    FRUIT_TYPES,
    QUALITY_TYPES,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
)
from app_modules.image_processing import (
    get_u2net_session,
    segment_fruit_image,
    load_image_to_pil,
)
from app_modules.feature_extraction import (
    extract_features_from_segmented,
    FEATURE_METADATA,
)
from app_modules.prediction import predict_fruit_quality
from app_modules.database import (
    save_prediction,
    get_all_predictions,
    clear_history,
    get_statistics,
)

# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Fruit Quality Grading System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS STYLING (Dark Neon Theme)
# --------------------------------------------------
# Hero image is optional — the dark theme does not require it.
_hero_img_path = PROJECT_ROOT / "assets" / "fruit-dashboard-hero.jpg"
_HERO_B64 = (
    base64.b64encode(_hero_img_path.read_bytes()).decode("ascii")
    if _hero_img_path.exists()
    else ""
)

css = """
<style>
    /* ============================================================
       DARK THEME — Fruit Quality Grading System
       Neon-green glowing borders, deep dark background
    ============================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* ── Keyframes ── */
    @keyframes borderGlow {
        0%   { box-shadow: 0 0 6px #00e676, 0 0 14px rgba(0,230,118,0.35); }
        50%  { box-shadow: 0 0 14px #00e676, 0 0 32px rgba(0,230,118,0.55), 0 0 56px rgba(0,230,118,0.18); }
        100% { box-shadow: 0 0 6px #00e676, 0 0 14px rgba(0,230,118,0.35); }
    }
    @keyframes subtleGlow {
        0%   { box-shadow: 0 0 4px rgba(0,230,118,0.25), 0 0 10px rgba(0,230,118,0.12); }
        50%  { box-shadow: 0 0 10px rgba(0,230,118,0.45), 0 0 24px rgba(0,230,118,0.20); }
        100% { box-shadow: 0 0 4px rgba(0,230,118,0.25), 0 0 10px rgba(0,230,118,0.12); }
    }
    @keyframes dotPulse {
        0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.70); }
        70%  { box-shadow: 0 0 0 7px rgba(0,230,118,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,230,118,0); }
    }

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8e4;
    }

    /* Dark app background */
    .stApp {
        background: #0a0f0a !important;
    }
    [data-testid="stAppViewContainer"] {
        background: #0a0f0a !important;
    }
    [data-testid="stHeader"] {
        background: rgba(10,15,10,0.85) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(0,230,118,0.15);
    }
    [data-testid="stToolbar"] { visibility: hidden; }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
        background: transparent;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        color: #69f0ae !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    p, li, label, span, div {
        color: #c8d8c8;
    }

    /* ── Streamlit native widgets — dark overrides ── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stRadio"] > div,
    [data-testid="stFileUploader"] > div {
        background: #111a11 !important;
        border-color: rgba(0,230,118,0.25) !important;
        color: #c8d8c8 !important;
        border-radius: 10px;
    }
    .stDataFrame, [data-testid="stDataFrame"] {
        background: #0d150d;
        border: 1px solid rgba(0,230,118,0.20);
        border-radius: 10px;
        animation: subtleGlow 4s ease-in-out infinite;
    }
    [data-testid="stMetric"] {
        background: #0e170e !important;
        border: 1px solid rgba(0,230,118,0.22) !important;
        border-radius: 10px !important;
        padding: 0.6rem !important;
        animation: subtleGlow 4s ease-in-out infinite;
    }
    [data-testid="stMetricLabel"] > div { color: #7aaa8a !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] > div { color: #69f0ae !important; font-weight: 800 !important; }
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #00c853, #69f0ae) !important;
    }
    [data-testid="stProgress"] {
        background: #1a2a1a !important;
        border-radius: 8px;
    }
    .stAlert { border-radius: 10px !important; }
    [data-testid="stAlert"][kind="info"]    { background: #0d1f0d !important; border-color: #1b5e20 !important; }
    [data-testid="stAlert"][kind="success"] { background: #0a1f12 !important; border-color: #00c853 !important; }
    [data-testid="stAlert"][kind="warning"] { background: #1f1800 !important; border-color: #f9a825 !important; }
    [data-testid="stAlert"][kind="error"]   { background: #1f0a0a !important; border-color: #c62828 !important; }

    /* ── Caption / small text ── */
    .stCaption, [data-testid="stCaptionContainer"] { color: #5a7a5a !important; }

    /* ── Hero Banner ── */
    .hero-card {
        background: linear-gradient(135deg, #071407 0%, #0d2010 55%, #112a14 100%);
        border-radius: 16px;
        padding: 0;
        color: #ffffff;
        border: 1px solid rgba(0,230,118,0.30);
        box-shadow:
            0 0 0 1px rgba(0,230,118,0.20),
            0 0 24px rgba(0,230,118,0.20),
            0 0 60px rgba(0,230,118,0.08);
        margin-bottom: 24px;
        overflow: hidden;
        animation: borderGlow 3.5s ease-in-out infinite;
    }
    .hero-layout {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(260px, 0.55fr);
        min-height: 260px;
    }
    .hero-copy { padding: 32px 36px; }
    .hero-image-wrap {
        min-height: 260px;
        background: #0a1a0a;
        overflow: hidden;
    }
    .hero-image-wrap img {
        width: 100%; height: 100%; min-height: 260px;
        display: block; object-fit: cover; object-position: center;
        opacity: 0.75;
        filter: brightness(0.85) saturate(1.2);
    }
    @media (max-width: 900px) {
        .hero-layout { grid-template-columns: 1fr; }
        .hero-copy { padding: 26px 24px; }
        .hero-image-wrap, .hero-image-wrap img { min-height: 200px; max-height: 200px; }
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 800;
        color: #ffffff; margin: 0 0 8px 0; letter-spacing: -0.02em;
        text-shadow: 0 0 24px rgba(0,230,118,0.4);
    }
    .hero-subtitle {
        font-size: 1.05rem; color: #a5d6a7; margin: 0 0 16px 0; font-weight: 500;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,230,118,0.12);
        backdrop-filter: blur(8px);
        padding: 6px 14px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600; color: #69f0ae;
        border: 1px solid rgba(0,230,118,0.35);
        box-shadow: 0 0 10px rgba(0,230,118,0.20);
    }

    /* ── Stat Cards ── */
    .stat-card {
        background: linear-gradient(145deg, #0d1a0d, #111f11);
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid rgba(0,230,118,0.22);
        box-shadow: 0 0 10px rgba(0,230,118,0.10), inset 0 1px 0 rgba(0,230,118,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0,230,118,0.55);
        box-shadow: 0 0 22px rgba(0,230,118,0.30), 0 8px 24px rgba(0,0,0,0.4);
    }
    .stat-title {
        font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 0.07em; font-weight: 700; color: #4a7a5a; margin-bottom: 6px;
    }
    .stat-value {
        font-size: 1.85rem; font-weight: 800;
        color: #69f0ae; line-height: 1.2;
        text-shadow: 0 0 12px rgba(105,240,174,0.40);
    }
    .stat-subtext { font-size: 0.78rem; color: #4a6a4a; margin-top: 4px; }

    /* ── Result Cards ── */
    .result-card-fresh {
        background: linear-gradient(135deg, #071a0d 0%, #0d2514 100%);
        border: 2px solid #00e676;
        border-radius: 16px; padding: 28px; margin-top: 10px;
        box-shadow: 0 0 20px rgba(0,230,118,0.30), 0 0 50px rgba(0,230,118,0.10);
        animation: borderGlow 3s ease-in-out infinite;
    }
    .result-card-rotten {
        background: linear-gradient(135deg, #1a0707 0%, #251010 100%);
        border: 2px solid #ff5252;
        border-radius: 16px; padding: 28px; margin-top: 10px;
        box-shadow: 0 0 20px rgba(255,82,82,0.30), 0 0 50px rgba(255,82,82,0.10);
        animation: subtleGlow 3s ease-in-out infinite;
    }
    .quality-badge-fresh {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #000000; font-size: 1.1rem; font-weight: 800;
        padding: 8px 22px; border-radius: 30px; display: inline-block;
        letter-spacing: 0.04em;
        box-shadow: 0 0 16px rgba(0,230,118,0.50);
    }
    .quality-badge-rotten {
        background: linear-gradient(135deg, #c62828, #ff5252);
        color: #ffffff; font-size: 1.1rem; font-weight: 800;
        padding: 8px 22px; border-radius: 30px; display: inline-block;
        letter-spacing: 0.04em;
        box-shadow: 0 0 16px rgba(255,82,82,0.50);
    }

    /* ── Content Box ── */
    .content-box {
        background: #0d1a0d;
        border: 1px solid rgba(0,230,118,0.20);
        border-radius: 14px; padding: 22px; margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,230,118,0.08), inset 0 1px 0 rgba(0,230,118,0.06);
    }
    .content-box h4 { color: #69f0ae !important; }
    .content-box p, .content-box li { color: #9abaa0 !important; }

    /* ── Workflow Step Pills ── */
    .step-pill {
        background: #0d1a0d;
        border: 1px solid rgba(0,230,118,0.25);
        border-radius: 12px; padding: 16px 18px; text-align: center; height: 100%;
        box-shadow: 0 0 8px rgba(0,230,118,0.10);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .step-pill:hover {
        border-color: rgba(0,230,118,0.55);
        box-shadow: 0 0 18px rgba(0,230,118,0.25);
    }
    .step-num {
        background: linear-gradient(135deg, #00c853, #00e676);
        color: #000000; width: 28px; height: 28px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 0.85rem; margin-bottom: 8px;
        box-shadow: 0 0 10px rgba(0,230,118,0.50);
    }
    .step-title { font-weight: 700; color: #69f0ae; font-size: 0.95rem; margin-bottom: 4px; }
    .step-desc { font-size: 0.82rem; color: #6a8a6a; }

    /* ── Primary Buttons ── */
    .stButton > button {
        border-radius: 10px; font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00c853 0%, #00a040 100%);
        color: #000000 !important; border: none;
        box-shadow: 0 0 14px rgba(0,200,83,0.40);
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00e676 0%, #00c853 100%);
        box-shadow: 0 0 24px rgba(0,230,118,0.55);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(0,230,118,0.25) !important;
        color: #9abaa0 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(0,230,118,0.08) !important;
        border-color: rgba(0,230,118,0.50) !important;
        color: #69f0ae !important;
        box-shadow: 0 0 10px rgba(0,230,118,0.20);
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        background: transparent !important;
        border: 1px solid rgba(0,230,118,0.30) !important;
        color: #69f0ae !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(0,230,118,0.10) !important;
        box-shadow: 0 0 12px rgba(0,230,118,0.25);
    }

    /* ── Fruit category mini cards (home page) ── */
    .fruit-mini-card {
        background: #0d1a0d;
        border: 1px solid rgba(0,230,118,0.22);
        border-radius: 12px; padding: 12px 6px; text-align: center;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .fruit-mini-card:hover {
        border-color: rgba(0,230,118,0.55);
        box-shadow: 0 0 16px rgba(0,230,118,0.22);
    }

    /* ── Feature table header ── */
    .feature-table-header {
        background-color: #0d1a0d;
        font-weight: 700; color: #69f0ae;
    }

    /* ============================================================
       SIDEBAR — Dark + Neon Glow
    ============================================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060e06 0%, #080f08 100%) !important;
        border-right: 1px solid rgba(0,230,118,0.20) !important;
        box-shadow: 4px 0 24px rgba(0,230,118,0.08);
        min-width: 272px;
        max-width: 272px;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 0 0.85rem 1.5rem;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 { color: #69f0ae !important; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: #8aaa8a; }
    section[data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
        border-color: rgba(0,230,118,0.15);
        opacity: 1;
    }
    section[data-testid="stSidebar"] .stButton { margin-bottom: 0.18rem; }
    section[data-testid="stSidebar"] .stButton > button {
        min-height: 40px;
        justify-content: flex-start; text-align: left;
        padding: 0.52rem 0.75rem 0.52rem 0.85rem;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: #7aaa7a;
        font-size: 0.875rem; font-weight: 500;
        box-shadow: none; transition: all 0.14s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0,230,118,0.08);
        border-color: rgba(0,230,118,0.30);
        color: #69f0ae;
        box-shadow: 0 0 10px rgba(0,230,118,0.15);
        transform: none;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #00c853 0%, #00a040 100%);
        border-color: transparent;
        color: #000000 !important;
        font-weight: 700;
        box-shadow: 0 0 14px rgba(0,200,83,0.40), 0 2px 6px rgba(0,0,0,0.5);
    }
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #0d1a0d !important;
        border: 1px solid rgba(0,230,118,0.22) !important;
        border-radius: 8px !important;
        animation: subtleGlow 4s ease-in-out infinite;
    }

    /* Sidebar brand */
    .sb-brand {
        background: linear-gradient(135deg, #030a03 0%, #071407 60%, #0a1f0a 100%);
        border-radius: 0 0 16px 16px;
        padding: 22px 16px 18px;
        margin: 0 -0.85rem 14px;
        text-align: center;
        border-bottom: 1px solid rgba(0,230,118,0.30);
        box-shadow:
            0 6px 24px rgba(0,0,0,0.6),
            0 0 30px rgba(0,230,118,0.12),
            inset 0 -1px 0 rgba(0,230,118,0.20);
    }
    .sb-brand-icon {
        font-size: 2.2rem; line-height: 1; margin-bottom: 8px; display: block;
        filter: drop-shadow(0 0 10px rgba(0,230,118,0.60));
    }
    .sb-brand-title {
        color: #ffffff; font-size: 0.95rem; font-weight: 800;
        letter-spacing: -0.01em; margin-bottom: 6px; line-height: 1.25;
        text-shadow: 0 0 16px rgba(0,230,118,0.35);
    }
    .sb-brand-pill {
        display: inline-block;
        background: rgba(0,230,118,0.10);
        border: 1px solid rgba(0,230,118,0.30);
        color: #69f0ae;
        font-size: 0.65rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px;
        letter-spacing: 0.06em; text-transform: uppercase;
        box-shadow: 0 0 8px rgba(0,230,118,0.20);
    }

    /* Sidebar section label */
    .sb-section-label {
        font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.10em;
        color: rgba(0,230,118,0.45);
        margin: 10px 0 5px 2px; display: block;
    }

    /* Engine rows */
    .sb-engine-row {
        display: flex; align-items: center; gap: 10px;
        background: #0a140a;
        border: 1px solid rgba(0,230,118,0.18);
        border-radius: 9px; padding: 9px 11px; margin-bottom: 6px;
        box-shadow: 0 0 6px rgba(0,230,118,0.07), inset 0 1px 0 rgba(0,230,118,0.05);
    }
    .sb-engine-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #00e676; flex-shrink: 0;
        animation: dotPulse 2s ease-in-out infinite;
    }
    .sb-engine-info { flex: 1; }
    .sb-engine-name { font-size: 0.83rem; font-weight: 700; color: #c8e8c8; line-height: 1.2; }
    .sb-engine-sublabel { font-size: 0.69rem; color: #4a7a4a; font-weight: 500; }
    .sb-engine-badge {
        font-size: 0.62rem;
        background: rgba(0,230,118,0.10);
        color: #69f0ae;
        border: 1px solid rgba(0,230,118,0.28);
        border-radius: 10px; padding: 2px 7px; font-weight: 700; white-space: nowrap;
        box-shadow: 0 0 6px rgba(0,230,118,0.15);
    }

    /* Benchmark metric grid */
    .sb-metrics-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 2px;
    }
    .sb-metric-cell {
        background: #0a140a;
        border: 1px solid rgba(0,230,118,0.18);
        border-radius: 9px; padding: 9px 8px 8px; text-align: center;
        box-shadow: 0 0 6px rgba(0,230,118,0.07);
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .sb-metric-cell:hover {
        border-color: rgba(0,230,118,0.45);
        box-shadow: 0 0 14px rgba(0,230,118,0.20);
    }
    .sb-metric-val {
        font-size: 1.0rem; font-weight: 800; color: #69f0ae; line-height: 1.1;
        text-shadow: 0 0 8px rgba(105,240,174,0.35);
    }
    .sb-metric-lbl {
        font-size: 0.62rem; color: rgba(0,230,118,0.45); font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 3px;
    }

    /* Session status */
    .sb-session-active {
        background: rgba(0,200,83,0.07);
        border: 1px solid rgba(0,230,118,0.35);
        border-radius: 9px; padding: 9px 11px;
        display: flex; align-items: center; gap: 9px;
        box-shadow: 0 0 10px rgba(0,230,118,0.12);
        animation: subtleGlow 3s ease-in-out infinite;
    }
    .sb-session-idle {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 9px; padding: 9px 11px;
        display: flex; align-items: center; gap: 9px;
    }
    .sb-session-text { font-size: 0.80rem; font-weight: 700; color: #69f0ae; line-height: 1.2; }
    .sb-session-sub { font-size: 0.68rem; color: #3a7a3a; font-weight: 500; }
    .sb-session-text-idle { font-size: 0.80rem; font-weight: 500; color: #3a5a3a; }

    /* Sidebar footer */
    .sb-footer {
        text-align: center; padding: 8px 0 0;
        font-size: 0.65rem; color: #2a4a2a; line-height: 1.5;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "current_image_pil" not in st.session_state:
    st.session_state.current_image_pil = None
if "segmentation_data" not in st.session_state:
    st.session_state.segmentation_data = None
if "feature_dict" not in st.session_state:
    st.session_state.feature_dict = None
if "feature_df" not in st.session_state:
    st.session_state.feature_df = None
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# --------------------------------------------------
# HELPER: FIND DATASET SAMPLE IMAGES
# --------------------------------------------------
def get_sample_dataset_images():
    """Finds available sample images from the local dataset folder."""
    samples = []
    dataset_dir = PROJECT_ROOT / "Original Image"
    if dataset_dir.exists():
        for category_dir in sorted(dataset_dir.iterdir()):
            if category_dir.is_dir():
                first_img = next(category_dir.glob("*.jpg"), None)
                if first_img:
                    samples.append({
                        "category": category_dir.name,
                        "path": first_img,
                        "name": first_img.name
                    })
    return samples


# --------------------------------------------------
# SIDEBAR NAVIGATION & SYSTEM STATUS
# --------------------------------------------------
with st.sidebar:

    # ── Gradient brand header ──────────────────────
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-icon">🍎</span>
        <div class="sb-brand-title">Fruit Quality<br>Grading System</div>
        <span class="sb-brand-pill">AI · Vision · ML</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Page navigation ───────────────────────────
    pages = [
        "🏠 Home / Dashboard",
        "🔍 Fruit Quality Analysis",
        "🖼️ Image Segmentation",
        "📊 Feature Analysis",
        "ℹ️ Model Information",
        "📜 Prediction History",
    ]

    if "nav_page" not in st.session_state:
        st.session_state.nav_page = pages[0]
    if st.session_state.get("nav_jump") in pages:
        st.session_state.nav_page = st.session_state.pop("nav_jump")

    st.markdown('<span class="sb-section-label">📌 Navigation</span>', unsafe_allow_html=True)
    for page_name in pages:
        if st.button(
            page_name,
            key=f"nav_{page_name}",
            type="primary" if st.session_state.nav_page == page_name else "secondary",
            use_container_width=True,
        ):
            st.session_state.nav_page = page_name
            st.rerun()

    selected_page = st.session_state.nav_page

    st.markdown("---")

    # ── Session status ────────────────────────────
    st.markdown('<span class="sb-section-label">🔄 Session Status</span>', unsafe_allow_html=True)
    if st.session_state.get("analysis_complete") and st.session_state.get("prediction_result"):
        _res = st.session_state.prediction_result
        st.markdown(f"""
        <div class="sb-session-active">
            <span style="font-size:1.3rem; flex-shrink:0;">✅</span>
            <div>
                <div class="sb-session-text">{_res['fruit']} · {_res['quality']}</div>
                <div class="sb-session-sub">Analysis ready — check tabs</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sb-session-idle">
            <span style="font-size:1.2rem; flex-shrink:0;">💤</span>
            <div class="sb-session-text-idle">No image analysed yet</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── AI Engine status cards ────────────────────
    st.markdown('<span class="sb-section-label">⚡ AI Engines</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-engine-row">
        <div class="sb-engine-dot"></div>
        <div class="sb-engine-info">
            <div class="sb-engine-name">U²-Net</div>
            <div class="sb-engine-sublabel">Foreground segmentation</div>
        </div>
        <span class="sb-engine-badge">CPU · ONNX</span>
    </div>
    <div class="sb-engine-row">
        <div class="sb-engine-dot"></div>
        <div class="sb-engine-info">
            <div class="sb-engine-name">Random Forest</div>
            <div class="sb-engine-sublabel">Quality classifier</div>
        </div>
        <span class="sb-engine-badge">300 trees</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Benchmark metric grid (2 × 2) ────────────
    st.markdown('<span class="sb-section-label">📊 Benchmark</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-metrics-grid">
        <div class="sb-metric-cell">
            <div class="sb-metric-val">80.47%</div>
            <div class="sb-metric-lbl">Test Accuracy</div>
        </div>
        <div class="sb-metric-cell">
            <div class="sb-metric-val">80.24%</div>
            <div class="sb-metric-lbl">Macro F1</div>
        </div>
        <div class="sb-metric-cell">
            <div class="sb-metric-val">16</div>
            <div class="sb-metric-lbl">Classes</div>
        </div>
        <div class="sb-metric-cell">
            <div class="sb-metric-val">7</div>
            <div class="sb-metric-lbl">Features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Footer note ───────────────────────────────
    st.markdown("""
    <div class="sb-footer">
        Offline evaluation on a 640-image<br>stratified hold-out test split.
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# PAGE 1: HOME / DASHBOARD
# ==================================================
if selected_page == "🏠 Home / Dashboard":
    # Hero Card
    hero_image_path = PROJECT_ROOT / "assets" / "fruit-dashboard-hero.jpg"
    hero_image_data = base64.b64encode(hero_image_path.read_bytes()).decode("ascii")
    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-layout">
            <div class="hero-copy">
                <div class="hero-badge">🎓 Academic Project • Computer Vision & Machine Learning</div>
                <h1 class="hero-title">Fruit Quality Grading System</h1>
                <p class="hero-subtitle">
                    AI-powered fruit freshness analysis using U²-Net deep segmentation,
                    HSV colour metrics, GLCM surface texture extraction, and Random Forest classification.
                </p>
            </div>
            <div class="hero-image-wrap">
                <img src="data:image/jpeg;base64,{hero_image_data}"
                     alt="Fresh fruit quality grading illustration">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Fruit Categories</div>
            <div class="stat-value">8</div>
            <div class="stat-subtext">Apple, Banana, Grape, Guava, Jujube, Orange, Pomegranate, Strawberry</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Quality Classes</div>
            <div class="stat-value">2</div>
            <div class="stat-subtext">Binary Fresh / Rotten grading per fruit (16 total target classes)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Experimental Accuracy</div>
            <div class="stat-value">80.47%</div>
            <div class="stat-subtext">Evaluated on 640-sample test set (Random Forest 300 estimators)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-title">Macro F1-Score</div>
            <div class="stat-value">80.24%</div>
            <div class="stat-subtext">Balanced multi-class performance metric across all 16 classes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Workflow Overview Cards
    st.markdown("### 🔄 End-to-End Analysis Workflow")
    st.markdown("How this intelligent inspection system evaluates fruit freshness from a single camera photograph:")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
        <div class="step-pill">
            <div class="step-num">1</div>
            <div class="step-title">Image Acquisition</div>
            <div class="step-desc">Accepts high-resolution JPG or PNG photographs of fruit from top or side angles.</div>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="step-pill">
            <div class="step-num">2</div>
            <div class="step-title">U²-Net Segmentation</div>
            <div class="step-desc">Deep salient object detection strips backgrounds, shadows, and tables using an alpha mask.</div>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="step-pill">
            <div class="step-num">3</div>
            <div class="step-title">Feature Extraction</div>
            <div class="step-desc">Calculates 3 HSV colour metrics (Hue, Sat, Val) and 4 GLCM texture metrics (Contrast, Energy, etc.).</div>
        </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown("""
        <div class="step-pill">
            <div class="step-num">4</div>
            <div class="step-title">Quality Prediction</div>
            <div class="step-desc">Trained Random Forest classifier predicts fruit type and Fresh/Rotten status with confidence score.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Call to Action & Navigation
    col_cta1, col_cta2 = st.columns([2, 1])
    with col_cta1:
        st.markdown("""
        <div class="content-box">
            <h4>Ready to test fruit freshness?</h4>
            <p style="font-size: 0.95rem; margin-bottom: 16px;">
                Upload any photo of an apple, banana, grape, guava, jujube, orange, pomegranate, or strawberry,
                or choose from the built-in sample dataset images to see segmentation, feature extraction, and prediction in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_cta2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Start Fruit Quality Analysis", type="primary", use_container_width=True):
            st.session_state["nav_jump"] = "🔍 Fruit Quality Analysis"
            st.info("👈 Please select **🔍 Fruit Quality Analysis** from the sidebar navigation to begin.")

    # Fruit Categories Overview Grid
    st.markdown("### 🍓 Supported Fruit Categories")
    f_cols = st.columns(8)
    fruit_emojis = {
        "Apple": "🍎", "Banana": "🍌", "Grape": "🍇", "Guava": "🍈",
        "Jujube": "🫐", "Orange": "🍊", "Pomegranate": "🪴", "Strawberry": "🍓"
    }
    for col, fruit in zip(f_cols, FRUIT_TYPES):
        with col:
            st.markdown(f"""
            <div class="fruit-mini-card">
                <div style="font-size: 1.8rem; margin-bottom: 4px; filter: drop-shadow(0 0 8px rgba(0,230,118,0.4));">{fruit_emojis.get(fruit, '🍎')}</div>
                <div style="font-weight: 700; color: #69f0ae; font-size: 0.85rem;">{fruit}</div>
                <div style="font-size: 0.70rem; color: #3a6a3a;">Fresh / Rotten</div>
            </div>
            """, unsafe_allow_html=True)


# ==================================================
# PAGE 2: FRUIT QUALITY ANALYSIS (MAIN FUNCTIONALITY)
# ==================================================
elif selected_page == "🔍 Fruit Quality Analysis":
    st.markdown("## 🔍 Fruit Quality Analysis")
    st.markdown("Upload a fruit image to segment the foreground, extract HSV and GLCM features, and determine freshness.")

    col_up1, col_up2 = st.columns([1, 1])

    with col_up1:
        st.markdown("#### 1. Select or Upload Image")
        source_mode = st.radio(
            "Image Source",
            ["📤 Upload My Own Image", "📁 Select from Dataset Samples"],
            horizontal=True
        )

        selected_image_pil = None
        selected_filename = "uploaded_fruit.jpg"

        if source_mode == "📤 Upload My Own Image":
            uploaded_file = st.file_uploader(
                "Upload a fruit photograph (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="For best results, place the fruit centered with reasonable illumination."
            )
            if uploaded_file is not None:
                try:
                    selected_image_pil = load_image_to_pil(uploaded_file)
                    selected_filename = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading uploaded image: {e}")

        else:
            sample_list = get_sample_dataset_images()
            if sample_list:
                sample_options = [f"{s['category']} - {s['name']}" for s in sample_list]
                chosen_sample_idx = st.selectbox(
                    "Choose a dataset image:",
                    range(len(sample_options)),
                    format_func=lambda i: sample_options[i]
                )
                chosen_sample = sample_list[chosen_sample_idx]
                try:
                    selected_image_pil = Image.open(chosen_sample["path"]).convert("RGB")
                    selected_filename = chosen_sample["name"]
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
            else:
                st.warning("No sample images found in 'Original Image' folder. Please upload an image.")

        # Display image preview
        if selected_image_pil is not None:
            st.image(
                selected_image_pil,
                caption=f"Selected: {selected_filename} ({selected_image_pil.size[0]}×{selected_image_pil.size[1]} px)",
                use_container_width=True
            )

            # Analyze Button
            analyze_clicked = st.button("🔬 Analyze Fruit Freshness", type="primary", use_container_width=True)

    with col_up2:
        st.markdown("#### 2. Analysis & Grading Output")

        if selected_image_pil is not None and ('analyze_clicked' in locals() and analyze_clicked):
            with st.spinner("Processing image through U²-Net segmentation and feature pipeline..."):
                start_time = time.time()
                try:
                    # 1. Load model and U2-Net session
                    model = load_trained_model()
                    session = get_u2net_session()

                    # 2. Segment image
                    seg_data = segment_fruit_image(selected_image_pil, session=session)

                    # 3. Extract 7 features
                    feat_dict, feat_df = extract_features_from_segmented(
                        seg_data["rgb_image"],
                        seg_data["mask"]
                    )

                    # 4. Predict fruit and quality
                    pred_res = predict_fruit_quality(model, feat_df)

                    elapsed = round(time.time() - start_time, 2)

                    # 5. Save to local SQLite database
                    save_prediction(
                        filename=selected_filename,
                        fruit=pred_res["fruit"],
                        quality=pred_res["quality"],
                        class_label=pred_res["class_label"],
                        probability_score=pred_res["probability_score"],
                        feature_dict=feat_dict
                    )

                    # 6. Save in session state for other tabs
                    st.session_state.analysis_complete = True
                    st.session_state.current_image_name = selected_filename
                    st.session_state.current_image_pil = selected_image_pil
                    st.session_state.segmentation_data = seg_data
                    st.session_state.feature_dict = feat_dict
                    st.session_state.feature_df = feat_df
                    st.session_state.prediction_result = pred_res

                    st.success(f"Analysis completed in {elapsed}s!")

                except Exception as err:
                    st.session_state.analysis_complete = False
                    st.error(f"Analysis error: {str(err)}")

        # Display results if available in session state
        if st.session_state.analysis_complete and st.session_state.prediction_result is not None:
            res = st.session_state.prediction_result
            is_fresh = (res["quality"] == "Fresh")

            card_class = "result-card-fresh" if is_fresh else "result-card-rotten"
            badge_class = "quality-badge-fresh" if is_fresh else "quality-badge-rotten"
            prob_pct = res["probability_percent"]

            # Visual Result Card
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 0.82rem; font-weight: 700; color: #7aaa8a; text-transform: uppercase; letter-spacing: 0.08em;">
                        Classification Result
                    </span>
                    <span class="{badge_class}">
                        {res['status_icon']} {res['quality'].upper()}
                    </span>
                </div>
                <div style="font-size: 2.4rem; font-weight: 800; color: #e2f5e2; margin: 4px 0 10px 0; text-shadow: 0 0 20px rgba(0,230,118,0.30);">
                    {res['fruit']}
                </div>
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; color: #7aaa8a; margin-bottom: 4px;">
                        <span>Model Probability Score:</span>
                        <span style="color: {res['status_color']}; font-weight: 800; text-shadow: 0 0 10px currentColor;">{prob_pct}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Streamlit native progress bar for calibrated score display
            st.progress(float(res["probability_score"]))
            st.caption("ℹ️ Probability score reflects Random Forest ensemble tree agreement.")

            # Top Classes Distribution
            st.markdown("##### 🏆 Top Prediction Candidates")
            top_df = pd.DataFrame(res["top_classes"])[["class_label", "fruit", "quality", "probability_percent"]]
            top_df.columns = ["Target Class", "Fruit", "Quality", "Probability (%)"]
            st.dataframe(
                top_df.style.format({"Probability (%)": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )

            # Quick navigation links
            st.markdown("""
            💡 **Next steps:**
            - Visit **🖼️ Image Segmentation** to inspect the isolated fruit surface.
            - Visit **📊 Feature Analysis** to examine the exact calculated HSV and GLCM numbers.
            """)

        else:
            st.info("👈 Upload or select an image on the left, then click **🔬 Analyze Fruit Freshness** to run inspection.")


# ==================================================
# PAGE 3: IMAGE SEGMENTATION
# ==================================================
elif selected_page == "🖼️ Image Segmentation":
    st.markdown("## 🖼️ U²-Net Image Segmentation")
    st.markdown("Inspection of the background removal and alpha masking pipeline matching the model training.")

    if not st.session_state.analysis_complete or st.session_state.segmentation_data is None:
        st.warning("⚠️ No active image analysis found. Please run an analysis on the **🔍 Fruit Quality Analysis** page first.")
    else:
        seg = st.session_state.segmentation_data
        orig = st.session_state.current_image_pil

        # Side-by-side comparison
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown("#### 1. Original Image")
            st.image(orig, caption="Raw photograph input", use_container_width=True)

        with col_s2:
            st.markdown("#### 2. U²-Net Mask (Alpha)")
            st.image(seg["mask_pil"], caption="Binary foreground mask (224×224)", use_container_width=True)

        with col_s3:
            st.markdown("#### 3. Segmented Fruit")
            st.image(seg["preview_white_bg"], caption="Isolated fruit on clean background", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Foreground Statistics
        st.markdown("### 📐 Foreground Surface Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Detected Fruit Pixels", f"{seg['foreground_pixels']:,} px")
        with m2:
            st.metric("Image Area Coverage", f"{seg['foreground_ratio']:.2f}%")
        with m3:
            st.metric("Standardized Dimension", "224 × 224 px")

        st.markdown("<br>", unsafe_allow_html=True)

        # Educational Explanation Card
        st.markdown("""
        <div class="content-box">
            <h4>💡 Why is Image Segmentation Critical for Quality Grading?</h4>
            <p style="color: #334155; line-height: 1.6;">
                In real-world agricultural photographs, fruits are often placed on wooden tables, plates, conveyor belts,
                or held by hands. Background clutter introduces extraneous colours (brown wood, shadows, ambient lighting)
                that distort colour averages and texture calculations.
            </p>
            <ul style="color: #334155; line-height: 1.6;">
                <li><strong>Eliminates Environmental Noise:</strong> Strips background pixels entirely so only true fruit skin is measured.</li>
                <li><strong>Shadow Invariance:</strong> U²-Net's salient deep architecture distinguishes object cast shadows from the fruit body.</li>
                <li><strong>Consistent Geometric Sampling:</strong> Resizing to 224 × 224 standardizes texture spatial frequencies across all fruit types.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# PAGE 4: FEATURE ANALYSIS
# ==================================================
elif selected_page == "📊 Feature Analysis":
    st.markdown("## 📊 Feature Extraction & Numerical Analysis")
    st.markdown("Detailed breakdown of the exact 7 HSV colour and GLCM texture values supplied to the classifier.")

    if not st.session_state.analysis_complete or st.session_state.feature_dict is None:
        st.warning("⚠️ No active image analysis found. Please run an analysis on the **🔍 Fruit Quality Analysis** page first.")
    else:
        feats = st.session_state.feature_dict

        # Feature Summary Grid
        st.markdown("### 🧮 Calculated Numerical Values")

        col_f1, col_f2 = st.columns([1, 1])

        with col_f1:
            st.markdown("#### 🎨 Colour Features (HSV)")
            st.markdown("""
            HSV (Hue, Saturation, Value) represents human perception of colour, isolating chromatic tone from surface luminance.
            """)
            c_hsv1, c_hsv2, c_hsv3 = st.columns(3)
            with c_hsv1:
                st.metric("Mean Hue", f"{feats['mean_hue']:.2f}", help="Dominant colour tone (0-179 in OpenCV)")
            with c_hsv2:
                st.metric("Mean Saturation", f"{feats['mean_saturation']:.2f}", help="Colour vividness and purity (0-255)")
            with c_hsv3:
                st.metric("Mean Brightness", f"{feats['mean_brightness']:.2f}", help="Surface luminance (0-255)")

        with col_f2:
            st.markdown("#### 🔬 Texture Features (GLCM)")
            st.markdown("""
            Grey-Level Co-occurrence Matrix (GLCM) analyzes spatial relationships between horizontally adjacent foreground pixel pairs.
            """)
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric("Contrast", f"{feats['contrast']:.4f}", help="Local intensity variation")
            with t2:
                st.metric("Homogeneity", f"{feats['homogeneity']:.4f}", help="Surface smoothness")
            with t3:
                st.metric("Energy", f"{feats['energy']:.4f}", help="Texture regularity")
            with t4:
                st.metric("Correlation", f"{feats['correlation']:.4f}", help="Linear dependency of pixel pairs")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabular View with Descriptions
        st.markdown("### 📋 Features Table & Physical Role")
        rows = []
        for col_name in FEATURE_COLUMNS:
            val = feats[col_name]
            meta = FEATURE_METADATA.get(col_name, {})
            rows.append({
                "Feature": meta.get("name", col_name),
                "Category": meta.get("category", "General"),
                "Calculated Value": f"{val:.4f}",
                "Physical Measurement": meta.get("description", ""),
                "Role in Freshness Grading": meta.get("role", "")
            })

        table_df = pd.DataFrame(rows)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature Importance Reference from Training Run
        st.markdown("### 🌟 Model Feature Importance (Trained Benchmark)")
        feat_imp_path = PROJECT_ROOT / "segmented_feature_importance.csv"
        if feat_imp_path.exists():
            imp_df = pd.read_csv(feat_imp_path)
            imp_df["Importance (%)"] = (imp_df["Importance"] * 100).round(2)
            st.bar_chart(
                imp_df.set_index("Feature")["Importance (%)"],
                use_container_width=True
            )
            st.caption("Feature importance ranking recorded during model training on 3,200 segmented samples.")


# ==================================================
# PAGE 5: MODEL INFORMATION
# ==================================================
elif selected_page == "ℹ️ Model Information":
    st.markdown("## ℹ️ Model Architecture & Evaluation Benchmark")
    st.markdown("Comprehensive specifications of the trained Random Forest classifier and experimental evaluation.")

    # Model Architecture Details
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class="content-box">
            <h4>🌲 Random Forest Classifier</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Algorithm</td>
                    <td style="color: #1b5e20; font-weight: 700;">Random Forest (Ensemble Trees)</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Number of Estimators</td>
                    <td style="color: #1e293b; font-weight: 600;">300 Decision Trees</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Class Weighting</td>
                    <td style="color: #1e293b; font-weight: 600;">balanced_subsample</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Random State</td>
                    <td style="color: #1e293b; font-weight: 600;">42 (Reproducible)</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Input Features</td>
                    <td style="color: #1e293b; font-weight: 600;">7 Numerical Columns</td>
                </tr>
                <tr style="height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Output Classes</td>
                    <td style="color: #1e293b; font-weight: 600;">16 Classes (8 Fruits × 2 States)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="content-box">
            <h4>📈 Recorded Test Benchmark Results</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.92rem;">
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Overall Test Accuracy</td>
                    <td style="color: #2e7d32; font-weight: 800; font-size: 1.1rem;">80.47%</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Macro Average F1-Score</td>
                    <td style="color: #2e7d32; font-weight: 800; font-size: 1.1rem;">80.24%</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Weighted Average F1-Score</td>
                    <td style="color: #1e293b; font-weight: 700;">80.24%</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Test Set Size</td>
                    <td style="color: #1e293b; font-weight: 600;">640 images (20% stratified)</td>
                </tr>
                <tr style="height: 36px;">
                    <td style="font-weight: 600; color: #475569;">Total Segmented Dataset</td>
                    <td style="color: #1e293b; font-weight: 600;">3,200 images (200 per class)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Classification Report Table
    st.markdown("### 📑 Detailed Class-by-Class Evaluation Report")
    report_csv = PROJECT_ROOT / "segmented_classification_report.csv"
    if report_csv.exists():
        rep_df = pd.read_csv(report_csv, index_col=0)
        # Separate classes from aggregate rows
        class_rows = [c for c in rep_df.index if "_" in str(c)]
        class_sub = rep_df.loc[class_rows].copy()
        for col in ["precision", "recall", "f1-score"]:
            class_sub[col] = (class_sub[col] * 100).round(2)
        class_sub["support"] = class_sub["support"].astype(int)
        class_sub.columns = ["Precision (%)", "Recall (%)", "F1-Score (%)", "Support (Test Count)"]
        st.dataframe(class_sub, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion Matrix Visualization
    st.markdown("### 🔲 Evaluation Confusion Matrix")
    cm_img_path = PROJECT_ROOT / "segmented_confusion_matrix.png"
    if cm_img_path.exists():
        st.image(str(cm_img_path), caption="Normalized confusion matrix on 640-sample test split", use_container_width=True)


# ==================================================
# PAGE 6: PREDICTION HISTORY
# ==================================================
elif selected_page == "📜 Prediction History":
    st.markdown("## 📜 Prediction History")
    st.markdown("Local SQLite audit trail of all previous fruit grading analyses performed in this application.")

    # Statistics Bar
    stats = get_statistics()
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Analyses", stats["total_predictions"])
    with s2:
        st.metric("Fresh Fruits", stats["fresh_count"])
    with s3:
        st.metric("Rotten Fruits", stats["rotten_count"])
    with s4:
        st.metric("Freshness Ratio", f"{stats['fresh_ratio']}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter Controls
    col_fil1, col_fil2, col_fil3 = st.columns([2, 2, 3])
    with col_fil1:
        fruit_filter = st.selectbox("Filter by Fruit", ["All"] + FRUIT_TYPES)
    with col_fil2:
        quality_filter = st.selectbox("Filter by Quality", ["All"] + QUALITY_TYPES)

    # Fetch History DataFrame
    history_df = get_all_predictions(limit=200, fruit_filter=fruit_filter, quality_filter=quality_filter)

    if not history_df.empty:
        # Display formatted table
        display_df = history_df.copy()
        display_df["probability_score"] = (display_df["probability_score"] * 100).round(2).astype(str) + "%"

        st.dataframe(
            display_df[[
                "id", "timestamp", "filename", "fruit", "quality",
                "probability_score", "mean_hue", "mean_saturation",
                "mean_brightness", "contrast", "homogeneity", "energy", "correlation"
            ]],
            use_container_width=True,
            hide_index=True
        )

        # Export and Clear Controls
        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            csv_data = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download History as CSV",
                data=csv_data,
                file_name="fruit_grading_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_act2:
            confirm_clear = st.checkbox("Confirm clearing entire history")
            if st.button("🗑️ Clear History", type="secondary", disabled=not confirm_clear, use_container_width=True):
                if clear_history():
                    st.success("Prediction history cleared successfully.")
                    st.rerun()
                else:
                    st.error("Failed to clear database.")

    else:
        st.info("No prediction history recorded yet. Run an analysis on the **🔍 Fruit Quality Analysis** page to generate records.")


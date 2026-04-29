"""
RELIANCE Industries — AttentionGRU Forecast Dashboard
Cinematic stock-market aesthetic inspired by deep navy / amber glow / red-green tickers
All data built-in — no uploads required
Streamlit Cloud compatible (Altair only)
"""

import os, json, warnings, base64, io
from io import StringIO
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RELIANCE · AI Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  BUILT-IN DATA  — forecast CSV embedded directly
# ─────────────────────────────────────────────────────────────────────────────
FORECAST_CSV = """Date,Predicted_Close,Pct_vs_last
2026-03-10,1405.61,0.44
2026-03-11,1406.21,0.48
2026-03-12,1406.91,0.53
2026-03-13,1407.59,0.58
2026-03-16,1408.22,0.62
2026-03-17,1408.80,0.66
2026-03-18,1409.27,0.70
2026-03-19,1409.70,0.73
2026-03-20,1410.07,0.76
2026-03-23,1410.43,0.78
2026-03-24,1410.66,0.80
2026-03-25,1410.93,0.82
2026-03-26,1411.19,0.84
2026-03-27,1411.55,0.86
2026-03-30,1411.77,0.88
2026-03-31,1411.77,0.88
2026-04-01,1411.87,0.88
2026-04-02,1411.85,0.88
2026-04-03,1411.89,0.89
2026-04-06,1412.02,0.89
2026-04-07,1411.98,0.89
2026-04-08,1411.98,0.89
2026-04-09,1412.00,0.89
2026-04-10,1412.02,0.89
2026-04-13,1412.24,0.91
2026-04-14,1412.43,0.92
2026-04-15,1412.66,0.94
2026-04-16,1412.78,0.95
2026-04-17,1413.04,0.97
2026-04-20,1414.59,1.08
"""

METRICS_DATA = [
    {"Fold": "Fold 1 ★", "MAPE_pct": 4.942, "R2": 0.9560, "DirAcc": 54.43, "best": True},
    {"Fold": "Fold 2",   "MAPE_pct": 6.040, "R2": 0.8595, "DirAcc": 51.05, "best": False},
    {"Fold": "Fold 3",   "MAPE_pct": 3.742, "R2": 0.8028, "DirAcc": 50.11, "best": False},
    {"Fold": "Fold 4",   "MAPE_pct": 4.850, "R2": 0.9531, "DirAcc": 52.11, "best": False},
    {"Fold": "TEST",     "MAPE_pct": 3.164, "R2": 0.8799, "DirAcc": 50.45, "best": False},
]

MODEL_CONFIG = {
    "model_name": "AttentionGRU_v4",
    "sequence_len": 60,
    "n_features": 13,
    "feature_cols": ["Open","High","Low","Close","Volume","Return","HL_Ratio",
                     "OC_Ratio","Momentum_5","Momentum_10","Vol_10","MA_Ratio","Log_Volume"],
    "target_col": "Log_Return",
    "trained_on_fold": 1,
    "train_end_date": "2001-07-02",
    "architecture": {"gru1":160,"gru2":96,"attn_units":64,"dense1":64,"dense2":32,
                     "dropout":0.15,"l2":5e-6,"recurrent_dropout":0.05},
    "training": {"loss":"huber","optimizer":"adam","init_lr":5e-5,"peak_lr":0.0008,
                 "warmup_epochs":5,"batch_size":16,"max_epochs":150,"early_stopping_patience":35},
}

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (from the reference image)
# ─────────────────────────────────────────────────────────────────────────────
NAVY      = "#05071A"    # darkest bg — almost black-navy
NAVY2     = "#080C24"    # panel bg
NAVY3     = "#0D1230"    # card bg
NAVY4     = "#111840"    # elevated card
GLOW_TOP  = "#FF8C00"    # amber/orange glow (top-right)
GLOW_MID  = "#E05500"    # deep orange
LIME_GRN  = "#00FF88"    # bright green ticker
SOFT_GRN  = "#00CC66"    # mid green
TICKER_R  = "#FF3B5C"    # red ticker
SOFT_RED  = "#CC2244"    # mid red
ICE_BLUE  = "#4FC3F7"    # blue arrows / highlights
SILVER    = "#C8D6E5"    # primary text
SILVER2   = "#8A9BC0"    # secondary text
DIM_BLUE  = "#2A3560"    # grid / separator
FAINT     = "#1A2245"    # faint border
AMBER     = "#FFB347"    # gold/amber accent
WHITE     = "#FFFFFF"

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — cinematic dark with amber glow
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── Reset ── */
html,body,[class*="css"]{{
    background:{NAVY};
    color:{SILVER};
    font-family:'Rajdhani',sans-serif;
    font-size:15px;
}}
.block-container{{padding:0!important;max-width:100%!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stSidebar"]{{display:none;}}
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{NAVY};}}
::-webkit-scrollbar-thumb{{background:{DIM_BLUE};border-radius:2px;}}

/* ── Cinematic hero header ── */
.hero{{
    background: linear-gradient(135deg,
        {NAVY} 0%,
        {NAVY2} 30%,
        #0A0E28 55%,
        #160820 75%,
        #1A0A10 88%,
        #221006 100%);
    border-bottom:1px solid {DIM_BLUE};
    padding:2rem 2.5rem 1.75rem;
    position:relative;
    overflow:hidden;
}}
/* Amber glow ball (top-right) */
.hero::before{{
    content:'';
    position:absolute;
    top:-80px; right:-60px;
    width:420px; height:420px;
    background:radial-gradient(circle,
        {GLOW_TOP}55 0%,
        {GLOW_MID}33 30%,
        {GLOW_TOP}11 55%,
        transparent 70%);
    pointer-events:none;
    animation:glow-pulse 4s ease-in-out infinite;
}}
/* Secondary subtle blue glow */
.hero::after{{
    content:'';
    position:absolute;
    bottom:-40px; left:20%;
    width:300px; height:200px;
    background:radial-gradient(ellipse,{ICE_BLUE}18 0%,transparent 70%);
    pointer-events:none;
}}
@keyframes glow-pulse{{
    0%,100%{{opacity:1; transform:scale(1);}}
    50%{{opacity:.75; transform:scale(1.08);}}
}}

/* Ticker tape (scrolling numbers) */
.ticker-wrap{{
    overflow:hidden;
    background:{NAVY2};
    border-bottom:1px solid {DIM_BLUE};
    border-top:1px solid {DIM_BLUE};
    padding:.35rem 0;
    white-space:nowrap;
}}
.ticker-inner{{
    display:inline-block;
    animation:ticker-scroll 40s linear infinite;
    font-family:'Share Tech Mono',monospace;
    font-size:.78rem;
    letter-spacing:.04em;
}}
@keyframes ticker-scroll{{
    0%{{transform:translateX(0);}}
    100%{{transform:translateX(-50%);}}
}}
.tick-item{{
    display:inline-block;
    margin:0 2.5rem;
    color:{SILVER2};
}}
.tick-item .sym{{color:{AMBER};font-weight:700;margin-right:.4rem;}}
.tick-item .val{{color:{SILVER};}}
.tick-item .chg.up{{color:{LIME_GRN};}}
.tick-item .chg.dn{{color:{TICKER_R};}}

/* Brand / title */
.brand-tag{{
    font-family:'Orbitron',monospace;
    font-size:.65rem;
    color:{AMBER};
    letter-spacing:.35em;
    text-transform:uppercase;
    margin-bottom:.5rem;
    display:flex;
    align-items:center;
    gap:.6rem;
}}
.brand-tag .live-dot{{
    width:7px; height:7px;
    background:{LIME_GRN};
    border-radius:50%;
    box-shadow:0 0 8px {LIME_GRN};
    animation:live-pulse 1.8s ease-in-out infinite;
}}
@keyframes live-pulse{{
    0%,100%{{box-shadow:0 0 4px {LIME_GRN};}}
    50%{{box-shadow:0 0 14px {LIME_GRN}, 0 0 28px {LIME_GRN}66;}}
}}
.hero-title{{
    font-family:'Orbitron',monospace;
    font-size:2rem;
    font-weight:900;
    color:{WHITE};
    letter-spacing:.06em;
    text-transform:uppercase;
    margin:0 0 .3rem;
    text-shadow:0 0 30px {GLOW_TOP}88;
}}
.hero-title span{{
    background:linear-gradient(90deg,{GLOW_TOP},{AMBER},{LIME_GRN});
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}}
.hero-sub{{
    font-family:'Rajdhani',sans-serif;
    font-size:.9rem;
    color:{SILVER2};
    letter-spacing:.12em;
    text-transform:uppercase;
}}
.model-badge{{
    display:inline-flex; align-items:center; gap:.4rem;
    background:{NAVY4};
    border:1px solid {AMBER}55;
    color:{AMBER};
    font-family:'Share Tech Mono',monospace;
    font-size:.68rem;
    padding:.22rem .7rem;
    border-radius:3px;
    margin-left:1rem;
    vertical-align:middle;
    letter-spacing:.06em;
}}

/* Hero stats row */
.hero-stats{{
    display:flex; gap:2.5rem; margin-top:1.25rem; flex-wrap:wrap;
}}
.hstat{{display:flex; flex-direction:column; gap:.15rem;}}
.hstat .hl{{font-family:'Share Tech Mono',monospace; font-size:.6rem;
            color:{SILVER2}; letter-spacing:.12em; text-transform:uppercase;}}
.hstat .hv{{font-family:'Orbitron',monospace; font-size:1.05rem;
            font-weight:700; color:{WHITE};}}
.hstat .hv.g{{color:{LIME_GRN}; text-shadow:0 0 10px {LIME_GRN}66;}}
.hstat .hv.a{{color:{AMBER}; text-shadow:0 0 10px {AMBER}55;}}
.hstat .hv.r{{color:{TICKER_R}; text-shadow:0 0 10px {TICKER_R}55;}}
.hstat .hv.b{{color:{ICE_BLUE}; text-shadow:0 0 10px {ICE_BLUE}55;}}

/* ── Tab navigation ── */
.tab-bar{{
    background:{NAVY2};
    border-bottom:1px solid {DIM_BLUE};
    display:flex;
    padding:0 1.5rem;
}}
.tab-btn{{
    background:none; border:none;
    padding:.65rem 1.5rem;
    font-family:'Rajdhani',sans-serif;
    font-size:.85rem; font-weight:600;
    color:{SILVER2};
    cursor:pointer;
    letter-spacing:.1em;
    text-transform:uppercase;
    border-bottom:2px solid transparent;
    transition:.2s;
    position:relative;
}}
.tab-btn:hover{{color:{SILVER};}}
.tab-btn.active{{
    color:{AMBER};
    border-bottom:2px solid {AMBER};
}}
.tab-btn.active::after{{
    content:'';
    position:absolute;
    bottom:-1px; left:0; right:0;
    height:1px;
    background:linear-gradient(90deg,transparent,{AMBER}88,transparent);
}}

/* ── Section title ── */
.sec-title{{
    font-family:'Orbitron',monospace;
    font-size:.72rem; font-weight:700;
    color:{SILVER2};
    letter-spacing:.2em;
    text-transform:uppercase;
    margin:1.25rem 0 .8rem;
    display:flex; align-items:center; gap:.75rem;
}}
.sec-title::after{{
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,{DIM_BLUE},{NAVY});
}}
.sec-title.amber{{color:{AMBER};}}
.sec-title.green{{color:{SOFT_GRN};}}

/* ── Metric cards ── */
.mc-grid{{display:grid; grid-template-columns:repeat(4,1fr); gap:.85rem; margin-bottom:1.25rem;}}
.mc{{
    background:linear-gradient(145deg,{NAVY3},{NAVY4});
    border:1px solid {DIM_BLUE};
    border-radius:6px;
    padding:1rem 1.2rem;
    position:relative; overflow:hidden;
}}
.mc::before{{
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,{AMBER}66,transparent);
}}
.mc .lbl{{
    font-family:'Share Tech Mono',monospace;
    font-size:.58rem; color:{SILVER2};
    letter-spacing:.14em; text-transform:uppercase;
    margin-bottom:.4rem;
}}
.mc .val{{
    font-family:'Orbitron',monospace;
    font-size:1.45rem; font-weight:700;
    color:{WHITE}; line-height:1;
}}
.mc .val.g{{color:{LIME_GRN}; text-shadow:0 0 12px {LIME_GRN}55;}}
.mc .val.a{{color:{AMBER}; text-shadow:0 0 12px {AMBER}55;}}
.mc .val.r{{color:{TICKER_R}; text-shadow:0 0 12px {TICKER_R}55;}}
.mc .val.b{{color:{ICE_BLUE}; text-shadow:0 0 12px {ICE_BLUE}55;}}
.mc .sub{{font-size:.68rem; color:{SILVER2}; margin-top:.3rem; font-weight:500;}}
.mc .sub.g{{color:{SOFT_GRN};}}
.mc .sub.r{{color:{SOFT_RED};}}

/* Arrow indicator */
.arrow-up{{color:{LIME_GRN}; font-size:1.1em; text-shadow:0 0 8px {LIME_GRN};}}
.arrow-dn{{color:{TICKER_R}; font-size:1.1em; text-shadow:0 0 8px {TICKER_R};}}

/* ── Forecast table ── */
.fc-wrap{{
    background:linear-gradient(180deg,{NAVY3},{NAVY2});
    border:1px solid {DIM_BLUE};
    border-radius:6px; overflow:hidden;
}}
.fc-hdr{{
    display:grid; grid-template-columns:1.2fr 1fr 1fr;
    padding:.5rem .85rem;
    background:{NAVY4};
    border-bottom:1px solid {DIM_BLUE};
    font-family:'Share Tech Mono',monospace;
    font-size:.58rem; color:{SILVER2};
    letter-spacing:.12em; text-transform:uppercase;
}}
.fc-row{{
    display:grid; grid-template-columns:1.2fr 1fr 1fr;
    padding:.42rem .85rem;
    border-bottom:1px solid {FAINT};
    font-size:.8rem; font-weight:500;
    transition:background .12s;
}}
.fc-row:hover{{background:{NAVY4};}}
.fc-row:last-child{{border-bottom:none;}}
.fc-row .date{{
    font-family:'Share Tech Mono',monospace;
    font-size:.73rem; color:{SILVER2};
}}
.fc-row .price{{
    text-align:right;
    font-family:'Share Tech Mono',monospace;
    color:{SILVER}; font-size:.78rem;
}}
.fc-row .pct{{
    text-align:right;
    font-family:'Share Tech Mono',monospace;
    font-size:.75rem;
}}
.fc-row .pct.up{{color:{LIME_GRN}; text-shadow:0 0 6px {LIME_GRN}55;}}
.fc-row .pct.dn{{color:{TICKER_R}; text-shadow:0 0 6px {TICKER_R}55;}}

/* ── Architecture diagram ── */
.arch-flow{{display:flex; flex-direction:column; gap:2px;}}
.arch-row{{display:flex; align-items:center; gap:.75rem; margin-bottom:1px;}}
.arch-box{{
    font-family:'Share Tech Mono',monospace; font-size:.65rem; font-weight:700;
    padding:.3rem .75rem; border-radius:3px;
    min-width:105px; text-align:center;
    border:1px solid; letter-spacing:.05em;
}}
.arch-desc{{font-size:.75rem; font-weight:500; color:{SILVER2};}}
.arch-arrow{{
    font-size:.85rem; color:{DIM_BLUE};
    margin-left:50px; line-height:.5; margin-bottom:1px;
}}

/* ── HP table ── */
.hp-table{{
    background:{NAVY3}; border:1px solid {DIM_BLUE};
    border-radius:6px; overflow:hidden;
}}
.hp-row{{
    display:flex; justify-content:space-between; align-items:center;
    padding:.4rem .9rem; border-bottom:1px solid {FAINT};
    font-size:.78rem; font-weight:500;
}}
.hp-row:last-child{{border-bottom:none;}}
.hp-k{{color:{SILVER2}; letter-spacing:.05em;}}
.hp-v{{font-family:'Share Tech Mono',monospace; color:{AMBER}; font-size:.75rem;}}

/* ── Feature pills ── */
.pill{{
    display:inline-block;
    background:{NAVY4}; border:1px solid {DIM_BLUE};
    border-radius:3px; padding:.2rem .55rem;
    font-family:'Share Tech Mono',monospace; font-size:.65rem;
    color:{SILVER2}; margin:.15rem;
    letter-spacing:.04em;
}}

/* ── Change-log table ── */
.chlog-row{{
    display:grid; grid-template-columns:1fr .8fr .8fr 2fr;
    padding:.35rem .85rem; border-bottom:1px solid {FAINT};
    font-size:.72rem; font-weight:500;
}}
.chlog-row:last-child{{border-bottom:none;}}
.chlog-row .p{{color:{SILVER2};font-family:'Share Tech Mono',monospace;font-size:.68rem;}}
.chlog-row .o{{color:{TICKER_R}; font-family:'Share Tech Mono',monospace;font-size:.68rem;}}
.chlog-row .n{{color:{LIME_GRN}; font-family:'Share Tech Mono',monospace;font-size:.68rem;}}
.chlog-row .r{{color:{SILVER2}; font-style:italic;}}

/* ── Plot frame ── */
.plot-frame{{
    background:{NAVY3}; border:1px solid {DIM_BLUE};
    border-radius:6px; padding:.5rem;
    overflow:hidden; margin-bottom:.9rem;
}}
.plot-cap{{
    font-family:'Share Tech Mono',monospace;
    font-size:.57rem; color:{SILVER2};
    letter-spacing:.1em; text-transform:uppercase;
    text-align:center; margin-top:.35rem;
}}

/* ── Disclaimer ── */
.disc{{
    background:{GLOW_TOP}0D; border:1px solid {GLOW_TOP}33;
    border-radius:4px; padding:.6rem .9rem;
    font-size:.72rem; color:{AMBER};
    margin-top:1rem; font-weight:500;
    letter-spacing:.04em; line-height:1.6;
}}

/* ── Streamlit widget overrides ── */
div[data-testid="stButton"]>button{{
    background:linear-gradient(135deg,{NAVY3},{NAVY4});
    border:1px solid {DIM_BLUE};
    color:{SILVER}; font-family:'Rajdhani',sans-serif;
    font-weight:600; letter-spacing:.1em; text-transform:uppercase;
    transition:.2s;
}}
div[data-testid="stButton"]>button:hover{{
    border-color:{AMBER}88; color:{AMBER};
}}
div[data-testid="stButton"]>button[kind="primary"]{{
    border-color:{AMBER}88; color:{AMBER};
    background:linear-gradient(135deg,{NAVY3},{NAVY4});
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR  = os.path.join(_APP_DIR, "outputs", "plots")
MODEL_DIR = os.path.join(_APP_DIR, "outputs", "models")

def plot_img(path, cap=""):
    if os.path.exists(path):
        with open(path,"rb") as f:
            b = base64.b64encode(f.read()).decode()
        st.markdown(f"""<div class='plot-frame'>
          <img src='data:image/png;base64,{b}'
               style='width:100%;border-radius:3px;display:block;'>
          {"<div class='plot-cap'>"+cap+"</div>" if cap else ""}
        </div>""", unsafe_allow_html=True)

def dark_alt(chart, h=300):
    return (
        chart.properties(height=h)
        .configure(background="transparent",
                   view=alt.ViewConfig(stroke=DIM_BLUE))
        .configure_axis(gridColor=FAINT, domainColor=DIM_BLUE,
                        labelColor=SILVER2, titleColor=SILVER2,
                        labelFont="Share Tech Mono, monospace",
                        titleFont="Share Tech Mono, monospace",
                        labelFontSize=10, titleFontSize=10)
        .configure_legend(labelColor=SILVER2, titleColor=SILVER2,
                          labelFont="Share Tech Mono, monospace",
                          titleFont="Share Tech Mono, monospace",
                          strokeColor=DIM_BLUE)
        .configure_title(color=SILVER2, font="Share Tech Mono, monospace", fontSize=10)
    )

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD BUILT-IN DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_builtin_forecast():
    df = pd.read_csv(StringIO(FORECAST_CSV), parse_dates=["Date"])
    df["direction"] = df["Pct_vs_last"].apply(lambda x: "UP" if x >= 0 else "DN")
    return df

@st.cache_data
def load_reliance_csv():
    # Resolve paths relative to this script so they work on Streamlit Cloud,
    # locally, and in any working-directory context.
    _here = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(_here, "RELIANCE.csv"),           # ← repo root (primary)
        os.path.join(_here, "data", "RELIANCE.csv"),
        os.path.join(_here, "outputs", "data", "RELIANCE.csv"),
        "RELIANCE.csv",                                 # cwd fallback
        "data/RELIANCE.csv",
        "outputs/data/RELIANCE.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df = df[cols].apply(pd.to_numeric, errors="coerce")
            df["Volume"] = df["Volume"].replace(0, np.nan)
            df.dropna(inplace=True)
            return df
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising AttentionGRU model…")
def load_model():
    try:
        import tensorflow as tf
        class BahdanauAttention(tf.keras.layers.Layer):
            def __init__(self, units, **kw):
                super().__init__(**kw); self.units = units
                self.W = tf.keras.layers.Dense(units)
                self.V = tf.keras.layers.Dense(1)
            def call(self, h):
                s = self.V(tf.nn.tanh(self.W(h)))
                s -= tf.reduce_max(s, axis=1, keepdims=True)
                a  = tf.nn.softmax(s, axis=1)
                return tf.reduce_sum(a * h, axis=1), a
            def get_config(self):
                c = super().get_config(); c["units"] = self.units; return c

        mdl = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "final_model.keras"),
            custom_objects={"BahdanauAttention": BahdanauAttention})
        fsc = joblib.load(os.path.join(MODEL_DIR, "final_f_scaler.pkl"))
        tsc = joblib.load(os.path.join(MODEL_DIR, "final_t_scaler.pkl"))
        return mdl, fsc, tsc, True
    except Exception:
        return None, None, None, False

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer(df):
    df = df.copy()
    df["Return"]      = df["Close"].pct_change()
    df["Log_Return"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Ratio"]    = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Ratio"]    = (df["Close"] - df["Open"]) / df["Open"]
    df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Vol_10"]      = df["Return"].rolling(10).std()
    df["MA5"]         = df["Close"].rolling(5).mean()
    df["MA20"]        = df["Close"].rolling(20).mean()
    df["MA_Ratio"]    = df["MA5"] / df["MA20"]
    df.drop(columns=["MA5","MA20"], inplace=True)
    df["Log_Volume"]  = np.log(df["Volume"].clip(lower=1))
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ALL DATA AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
fc_df   = load_builtin_forecast()
df_raw  = load_reliance_csv()
df_feat = engineer(df_raw) if df_raw is not None else None
mdl, fsc, tsc, model_ok = load_model()

# Key stats from built-in data
LAST_CLOSE  = float(df_raw["Close"].iloc[-1])    if df_raw  is not None else 1400.00
LAST_DATE   = df_raw.index[-1].strftime("%d %b %Y") if df_raw is not None else "07 Mar 2026"
PRED_30     = float(fc_df["Predicted_Close"].iloc[-1])
PCT_30      = float(fc_df["Pct_vs_last"].iloc[-1])
DAYS_UP     = int((fc_df["direction"] == "UP").sum())
DAYS_DN     = 30 - DAYS_UP

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "tab" not in st.session_state:
    st.session_state.tab = "forecast"

# ─────────────────────────────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
arr30  = "▲" if PCT_30 >= 0 else "▼"
cls30  = "g" if PCT_30 >= 0 else "r"

st.markdown(f"""
<div class='hero'>
  <div class='brand-tag'>
    <span class='live-dot'></span>
    NSE · RELIANCE INDUSTRIES LIMITED · AI FORECAST ENGINE
  </div>
  <div style='display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap;'>
    <h1 class='hero-title'>RELIANCE <span>STOCK FORECAST</span></h1>
    <span class='model-badge'>AttentionGRU · v4</span>
  </div>
  <div class='hero-sub'>Walk-Forward · Bahdanau Attention · Huber Loss · Adam Optimizer</div>

  <div class='hero-stats'>
    <div class='hstat'>
      <span class='hl'>Last Close</span>
      <span class='hv'>₹{LAST_CLOSE:,.2f}</span>
    </div>
    <div class='hstat'>
      <span class='hl'>30-Day Target</span>
      <span class='hv {"g" if PCT_30>=0 else "r"}'>{arr30} ₹{PRED_30:,.2f}</span>
    </div>
    <div class='hstat'>
      <span class='hl'>Expected Return</span>
      <span class='hv {"g" if PCT_30>=0 else "r"}'>{arr30} {abs(PCT_30):.2f}%</span>
    </div>
    <div class='hstat'>
      <span class='hl'>Bullish / Bearish</span>
      <span class='hv'><span style='color:{LIME_GRN}'>{DAYS_UP}▲</span> <span style='color:{SILVER2}'>·</span> <span style='color:{TICKER_R}'>{DAYS_DN}▼</span></span>
    </div>
    <div class='hstat'>
      <span class='hl'>Test MAPE</span>
      <span class='hv g'>3.164%</span>
    </div>
    <div class='hstat'>
      <span class='hl'>Test R²</span>
      <span class='hv b'>0.8799</span>
    </div>
    <div class='hstat'>
      <span class='hl'>Dir. Accuracy</span>
      <span class='hv a'>50.45%</span>
    </div>
    <div class='hstat'>
      <span class='hl'>Best Fold</span>
      <span class='hv a'>#1 ★</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TICKER TAPE
# ─────────────────────────────────────────────────────────────────────────────
tickers = [
    ("RELIANCE","₹1,400","▲ +0.89%","up"),("NIFTY50","22,147","▲ +0.44%","up"),
    ("SENSEX","72,988","▲ +0.38%","up"),("RELIANCE 30D","₹1,414","▲ +1.08%","up"),
    ("TEST MAPE","3.164%","✓","up"),("R²","0.8799","Fold 1","up"),
    ("DIR ACC","50.45%","+0.45%","up"),("SEQ LEN","60 days","13 features","up"),
    ("GRU","160/96","Bahdanau","up"),("TRAIN END","2001-07-02","Fold #1","up"),
]
tape = " &nbsp;|&nbsp; ".join(
    f"<span class='tick-item'>"
    f"<span class='sym'>{t[0]}</span>"
    f"<span class='val'>{t[1]}</span> "
    f"<span class='chg {t[3]}'>{t[2]}</span>"
    f"</span>"
    for t in tickers
)
# Duplicate for seamless loop
st.markdown(f"""
<div class='ticker-wrap'>
  <div class='ticker-inner'>{tape}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{tape}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB NAV
# ─────────────────────────────────────────────────────────────────────────────
TABS = [
    ("forecast",  "📈  Forecast"),
    ("history",   "📉  Price History"),
    ("plots",     "🔬  Analysis Plots"),
    ("metrics",   "📊  Metrics"),
    ("arch",      "⚙   Architecture"),
]
tab_cols = st.columns(len(TABS) + 6)
for i, (key, label) in enumerate(TABS):
    active = st.session_state.tab == key
    if tab_cols[i].button(
        label, key=f"tb_{key}",
        type="primary" if active else "secondary"
    ):
        st.session_state.tab = key
        st.rerun()

st.markdown(f"<hr style='margin:0;border:none;border-top:1px solid {DIM_BLUE};'>",
            unsafe_allow_html=True)

TAB = st.session_state.tab

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: metric card
# ─────────────────────────────────────────────────────────────────────────────
def mc(col, lbl, val, sub="", vcls="", scls=""):
    col.markdown(f"""<div class='mc'>
      <div class='lbl'>{lbl}</div>
      <div class='val {vcls}'>{val}</div>
      {"<div class='sub "+scls+"'>"+sub+"</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
if TAB == "forecast":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)

    # ── Summary metric strip ──────────────────────────────────────────────────
    st.markdown(f"<div class='sec-title amber'>30-Day Forecast Summary</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    mc(c1, "Last Close",      f"₹{LAST_CLOSE:,.2f}", LAST_DATE, "", "")
    pct_cls = "g" if PCT_30>=0 else "r"
    arr = "▲" if PCT_30>=0 else "▼"
    c2.markdown(f"""<div class='mc'>
      <div class='lbl'>Day 30 Target</div>
      <div class='val {pct_cls}'>₹{PRED_30:,.2f}</div>
      <div class='sub {pct_cls}'>{arr} {abs(PCT_30):.2f}% vs today</div>
    </div>""", unsafe_allow_html=True)
    mc(c3, "Bullish Days", f"{DAYS_UP}", f"of 30 forecast days", "g", "")
    mc(c4, "Bearish Days", f"{DAYS_DN}", f"of 30 forecast days", "r", "")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    # ── Main chart + table ────────────────────────────────────────────────────
    left, right = st.columns([3, 1], gap="medium")

    with left:
        st.markdown(f"<div class='sec-title'>Price Trajectory — Historical + 30-Day Forecast</div>",
                    unsafe_allow_html=True)

        # Build chart data
        chart_rows = []
        if df_raw is not None:
            tail = df_raw["Close"].iloc[-120:].reset_index()
            tail.columns = ["Date","Price"]
            tail["Series"] = "Historical"; tail["Dir"] = "—"
            chart_rows.append(tail)

        fp = fc_df[["Date","Predicted_Close","direction"]].copy()
        fp.columns = ["Date","Price","Dir"]; fp["Series"] = "Forecast"
        chart_rows.append(fp)

        pdf = pd.concat(chart_rows, ignore_index=True)
        pdf["Date"] = pd.to_datetime(pdf["Date"])

        hist_line = alt.Chart(pdf[pdf["Series"]=="Historical"]).mark_line(
            color=ICE_BLUE, strokeWidth=1.6, opacity=.85
        ).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", title="")),
            y=alt.Y("Price:Q", axis=alt.Axis(title="Price (₹)", format=",.0f"),
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Price:Q", format=",.2f", title="₹")]
        )

        # Forecast band
        fc_band_df = fc_df.copy()
        fc_band_df["upper"] = fc_band_df["Predicted_Close"] * 1.012
        fc_band_df["lower"] = fc_band_df["Predicted_Close"] * 0.988
        band = alt.Chart(pd.concat([
            fc_band_df[["Date","upper"]].rename(columns={"upper":"Price"}),
            fc_band_df[["Date","lower"]].rename(columns={"lower":"Price"}).iloc[::-1]
        ])).mark_area(color=AMBER, opacity=.08).encode(x="Date:T", y="Price:Q")

        fc_line = alt.Chart(pdf[pdf["Series"]=="Forecast"]).mark_line(
            color=AMBER, strokeWidth=2.2, strokeDash=[5,2]
        ).encode(x="Date:T", y="Price:Q")

        fc_dots = alt.Chart(pdf[pdf["Series"]=="Forecast"]).mark_circle(size=55).encode(
            x="Date:T", y="Price:Q",
            color=alt.Color("Dir:N",
                scale=alt.Scale(domain=["UP","DN"], range=[LIME_GRN, TICKER_R]),
                legend=alt.Legend(title="Direction")),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Price:Q", format=",.2f", title="₹"), "Dir:N"]
        )

        st.altair_chart(dark_alt(band + hist_line + fc_line + fc_dots, h=340),
                        use_container_width=True)

        # Pct change bar chart
        st.markdown(f"<div class='sec-title'>Cumulative % Change vs Last Close</div>",
                    unsafe_allow_html=True)
        pct_df = fc_df[["Date","Pct_vs_last","direction"]].copy()
        pct_bars = alt.Chart(pct_df).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%d %b", title="")),
            y=alt.Y("Pct_vs_last:Q", axis=alt.Axis(title="% vs last close")),
            color=alt.Color("direction:N",
                scale=alt.Scale(domain=["UP","DN"], range=[LIME_GRN, TICKER_R]),
                legend=None),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Pct_vs_last:Q", format=".2f", title="Chg%"),
                     "direction:N"]
        )
        zero_r = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
            color=SILVER2, strokeDash=[4,3], opacity=.4
        ).encode(y="y:Q")
        st.altair_chart(dark_alt(pct_bars + zero_r, h=180), use_container_width=True)

    with right:
        st.markdown(f"<div class='sec-title'>30-Day Table</div>", unsafe_allow_html=True)
        st.markdown("<div class='fc-wrap'>", unsafe_allow_html=True)
        st.markdown("""<div class='fc-hdr'>
          <span>Date</span>
          <span style='text-align:right;display:block'>₹ Price</span>
          <span style='text-align:right;display:block'>Chg%</span>
        </div>""", unsafe_allow_html=True)
        for _, row in fc_df.iterrows():
            d   = pd.to_datetime(row["Date"]).strftime("%d %b")
            p   = f"{float(row['Predicted_Close']):,.2f}"
            pv  = float(row["Pct_vs_last"])
            ps  = f"+{pv:.2f}%" if pv >= 0 else f"{pv:.2f}%"
            dc  = "up" if row["direction"] == "UP" else "dn"
            arr = "▲" if row["direction"] == "UP" else "▼"
            st.markdown(f"""<div class='fc-row'>
                <span class='date'>{d}</span>
                <span class='price'>{p}</span>
                <span class='pct {dc}'>{arr} {ps}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Static forecast chart if available
        if os.path.exists(f"{PLOT_DIR}/12_future_forecast_30d.png"):
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            plot_img(f"{PLOT_DIR}/12_future_forecast_30d.png", "Training run forecast")

        st.markdown(f"""<div class='disc'>
          ⚠ NOT FINANCIAL ADVICE<br>
          Research prototype only. Reliability degrades rapidly beyond day 5.
          Past performance ≠ future results.</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "history":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)

    if df_raw is not None:
        # Summary metrics
        st.markdown(f"<div class='sec-title amber'>Market Data — RELIANCE Industries (NSE)</div>",
                    unsafe_allow_html=True)
        last_1y  = df_raw["Close"].iloc[-252] if len(df_raw)>=252 else df_raw["Close"].iloc[0]
        ret_1y   = (LAST_CLOSE / last_1y - 1) * 100
        all_high = df_raw["High"].max()
        all_low  = df_raw["Low"].min()
        avg_vol  = df_raw["Volume"].iloc[-30:].mean()

        h1,h2,h3,h4 = st.columns(4)
        mc(h1,"Current Price",      f"₹{LAST_CLOSE:,.2f}", LAST_DATE)
        mc(h2,"52W High",           f"₹{df_raw['High'].iloc[-252:].max():,.2f}", "52-week high", "a","")
        mc(h3,"52W Low",            f"₹{df_raw['Low'].iloc[-252:].min():,.2f}", "52-week low", "r","")
        mc(h4,"1Y Return",
           f"{'▲' if ret_1y>=0 else '▼'} {abs(ret_1y):.1f}%",
           "vs 1 year ago", "g" if ret_1y>=0 else "r","")

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        # Window selector using real Streamlit slider (no expander)
        hist_window = st.select_slider(
            "Chart window",
            options=[("1M", 22), ("3M", 66), ("6M", 130), ("1Y", 252),
                     ("3Y", 756), ("5Y", 1260), ("ALL", len(df_raw))],
            value=("1Y", 252),
            format_func=lambda x: x[0],
        )
        hist = df_raw.iloc[-hist_window[1]:].copy().reset_index()
        hist.columns = ["Date"] + list(hist.columns[1:])
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist["Dir"]  = (hist["Close"] >= hist["Open"]).map({True:"UP", False:"DN"})
        hist["MA20"] = hist["Close"].rolling(20).mean()
        hist["MA50"] = hist["Close"].rolling(50).mean()
        hist["Ret"]  = hist["Close"].pct_change() * 100

        # Price + MAs
        st.markdown(f"<div class='sec-title'>Closing Price & Moving Averages</div>",
                    unsafe_allow_html=True)
        close_l = alt.Chart(hist).mark_line(color=ICE_BLUE, strokeWidth=1.5).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %y", title="")),
            y=alt.Y("Close:Q", axis=alt.Axis(title="Price (₹)", format=",.0f"),
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Close:Q", format=",.2f", title="₹")]
        )
        ma20_l = alt.Chart(hist.dropna(subset=["MA20"])).mark_line(
            color=AMBER, strokeWidth=1.2, strokeDash=[4,2], opacity=.8
        ).encode(x="Date:T", y=alt.Y("MA20:Q"))
        ma50_l = alt.Chart(hist.dropna(subset=["MA50"])).mark_line(
            color=SOFT_RED, strokeWidth=1, strokeDash=[5,3], opacity=.7
        ).encode(x="Date:T", y=alt.Y("MA50:Q"))
        st.altair_chart(dark_alt(close_l + ma20_l + ma50_l, h=320),
                        use_container_width=True)

        lc1, lc2 = st.columns(2, gap="medium")

        with lc1:
            st.markdown(f"<div class='sec-title'>Volume</div>", unsafe_allow_html=True)
            vb = alt.Chart(hist).mark_bar(opacity=.78).encode(
                x=alt.X("Date:T", axis=alt.Axis(format="%b %y", title="")),
                y=alt.Y("Volume:Q", axis=alt.Axis(title="Volume")),
                color=alt.Color("Dir:N",
                    scale=alt.Scale(domain=["UP","DN"], range=[SOFT_GRN, SOFT_RED]),
                    legend=None),
                tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                         alt.Tooltip("Volume:Q", format=",.0f")]
            )
            st.altair_chart(dark_alt(vb, h=200), use_container_width=True)

        with lc2:
            st.markdown(f"<div class='sec-title'>Daily Return Distribution</div>",
                        unsafe_allow_html=True)
            rh = alt.Chart(hist.dropna(subset=["Ret"])).mark_bar(
                color=ICE_BLUE, opacity=.75
            ).encode(
                x=alt.X("Ret:Q", bin=alt.Bin(maxbins=55),
                         axis=alt.Axis(title="Daily Return (%)")),
                y=alt.Y("count()", axis=alt.Axis(title="Count")),
                tooltip=[alt.Tooltip("Ret:Q", bin=True, title="Return %"), "count()"]
            )
            mv = hist["Ret"].mean()
            mr = alt.Chart(pd.DataFrame({"x":[mv]})).mark_rule(
                color=AMBER, strokeDash=[4,3]
            ).encode(x="x:Q")
            st.altair_chart(dark_alt(rh + mr, h=200), use_container_width=True)

        # Stats table
        st.markdown(f"<div class='sec-title'>Summary Statistics</div>", unsafe_allow_html=True)
        stats = hist[["Open","High","Low","Close","Volume"]].describe().T
        st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

    else:
        st.error("RELIANCE.csv not found. Place `RELIANCE.csv` in the **same folder as app.py** (repo root) on Streamlit Cloud.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "plots":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title amber'>Training Run — All Analysis Plots</div>",
                unsafe_allow_html=True)

    ALL_PLOTS = [
        ("01_eda_overview.png",           "EDA OVERVIEW — Price · Volume · Distributions · Correlation"),
        ("04_fold_training_curves.png",   "Walk-Forward Training Curves — Loss & MAE (4 Folds)"),
        ("06_test_actual_vs_predicted.png","Actual vs Predicted — Unseen Test Set (2021–2026)"),
        ("09_val_vs_test.png",            "Validation vs Test Comparison"),
        ("07_residuals.png",              "Residuals — Test Set"),
        ("08_scatter.png",                "Actual vs Predicted Scatter (₹)"),
        ("10_attention_weights.png",      "Bahdanau Attention Weights — Top-5 Attended Days (Red)"),
        ("11_metrics_summary.png",        "Model Performance Summary — MAPE & Directional Accuracy"),
        ("12_future_forecast_30d.png",    "30-Day Price Forecast"),
    ]

    # Full-width: 0,1,2,3,4,7,8  |  side-by-side: 5,6
    i = 0
    while i < len(ALL_PLOTS):
        fn, cap = ALL_PLOTS[i]
        path = os.path.join(PLOT_DIR, fn)
        if i == 5:
            c1, c2 = st.columns(2, gap="small")
            with c1: plot_img(path, cap)
            if i+1 < len(ALL_PLOTS):
                fn2, cap2 = ALL_PLOTS[i+1]
                with c2: plot_img(os.path.join(PLOT_DIR, fn2), cap2)
            i += 2
        else:
            plot_img(path, cap)
            i += 1

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "metrics":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title amber'>Hold-Out Test Set Performance</div>",
                unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.columns(5)
    mc(t1,"Test MAPE",    "3.164%",  "Mean Absolute % Error", "g")
    mc(t2,"Test R²",      "0.8799",  "Variance explained",    "b")
    mc(t3,"Directional",  "50.45%",  "vs 50% random baseline","a")
    mc(t4,"Edge",         "+0.45%",  "over random baseline",  "g")
    mc(t5,"Best Fold",    "#1",      "Selected by min val_loss","a")

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title'>Walk-Forward Fold Comparison</div>",
                unsafe_allow_html=True)

    mdf = pd.DataFrame(METRICS_DATA)
    fold_df = mdf[mdf["Fold"]!="TEST"].copy()
    fold_df["Color"] = fold_df["best"].apply(lambda x: AMBER if x else ICE_BLUE)

    ch1, ch2, ch3 = st.columns(3, gap="medium")

    with ch1:
        b = alt.Chart(fold_df).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y("MAPE_pct:Q", axis=alt.Axis(title="MAPE %")),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip("MAPE_pct:Q", format=".3f", title="MAPE %")]
        ).properties(title="MAPE %")
        st.altair_chart(dark_alt(b, h=240), use_container_width=True)

    with ch2:
        b2 = alt.Chart(fold_df).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y("R2:Q", axis=alt.Axis(title="R²"),
                    scale=alt.Scale(domain=[0,1])),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip("R2:Q", format=".4f", title="R²")]
        ).properties(title="R² Score")
        st.altair_chart(dark_alt(b2, h=240), use_container_width=True)

    with ch3:
        r50 = alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(
            color=TICKER_R, strokeDash=[5,3], opacity=.7
        ).encode(y="y:Q")
        b3 = alt.Chart(fold_df).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y("DirAcc:Q", axis=alt.Axis(title="Dir. Acc %"),
                    scale=alt.Scale(domain=[45,60])),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip("DirAcc:Q", format=".2f", title="Dir. Acc %")]
        ).properties(title="Directional Accuracy %")
        st.altair_chart(dark_alt(b3 + r50, h=240), use_container_width=True)

    st.markdown(f"<div class='sec-title'>Full Metrics Table</div>", unsafe_allow_html=True)
    disp = mdf[["Fold","MAPE_pct","R2","DirAcc"]].copy()
    disp.columns = ["Fold","MAPE %","R²","Dir. Acc %"]
    st.dataframe(
        disp.style.format({"MAPE %":"{:.3f}","R²":"{:.4f}","Dir. Acc %":"{:.2f}"}),
        use_container_width=True, height=210
    )

    plot_img(f"{PLOT_DIR}/11_metrics_summary.png", "Model Performance Summary")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif TAB == "arch":
    st.markdown("<div style='padding:1.25rem 1.5rem 0;'>", unsafe_allow_html=True)

    arch = MODEL_CONFIG["architecture"]
    tr   = MODEL_CONFIG["training"]

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown(f"<div class='sec-title amber'>Model Architecture</div>",
                    unsafe_allow_html=True)

        LAYERS = [
            ("INPUT",     f"(60 × 13) — OHLCV + 8 engineered features",          ICE_BLUE),
            ("GRU 1",     f"{arch['gru1']} units · return_sequences=True",         SOFT_GRN),
            ("LAYERNORM", "Normalise activations",                                  SILVER2),
            ("GRU 2",     f"{arch['gru2']} units · return_sequences=True",         SOFT_GRN),
            ("LAYERNORM", "Normalise activations",                                  SILVER2),
            ("ATTENTION", f"Bahdanau additive · {arch['attn_units']} units",       AMBER),
            ("DENSE 1",   f"{arch['dense1']} units · ReLU · Dropout({arch['dropout']})", ICE_BLUE),
            ("DENSE 2",   f"{arch['dense2']} units · ReLU · Dropout({arch['dropout']*0.5:.3f})", ICE_BLUE),
            ("OUTPUT",    "1 unit → Log-Return → inverse_transform → ₹",          LIME_GRN),
        ]

        for i, (name, desc, color) in enumerate(LAYERS):
            st.markdown(f"""
            <div class='arch-row'>
              <div class='arch-box' style='color:{color};border-color:{color}44;background:{color}0D;'>
                {name}
              </div>
              <div class='arch-desc'>{desc}</div>
            </div>
            {"<div class='arch-arrow'>↓</div>" if i < len(LAYERS)-1 else ""}
            """, unsafe_allow_html=True)

        st.markdown(f"<div class='sec-title' style='margin-top:1.5rem'>Input Features</div>",
                    unsafe_allow_html=True)
        pills = "".join([f"<span class='pill'>{f}</span>"
                         for f in MODEL_CONFIG["feature_cols"]])
        st.markdown(f"<div style='margin-bottom:.5rem'>{pills}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""<div style='font-family:Share Tech Mono,monospace;font-size:.65rem;
            color:{SILVER2};line-height:1.9;margin-top:.5rem;'>
          TARGET&nbsp;&nbsp;&nbsp;: <span style='color:{AMBER}'>Log_Return</span><br>
          FEAT SC  : RobustScaler (fit on train fold only)<br>
          TGT SC   : StandardScaler (fit on train fold only)<br>
          BEST FOLD: <span style='color:{AMBER}'>#1 ★</span> — selected by minimum val_loss
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='sec-title amber'>Hyperparameters</div>",
                    unsafe_allow_html=True)

        HP = [
            ("GRU Units",            f"{arch['gru1']} / {arch['gru2']}"),
            ("Attention Units",      str(arch["attn_units"])),
            ("Dense Units",          f"{arch['dense1']} / {arch['dense2']}"),
            ("Dropout",              str(arch["dropout"])),
            ("Recurrent Dropout",    str(arch["recurrent_dropout"])),
            ("L2 Regularisation",    str(arch["l2"])),
            ("Kernel Init",          "glorot_uniform"),
            ("Loss Function",        tr["loss"].upper()),
            ("Optimiser",            "Adam · clipnorm=1.0"),
            ("Init LR",              str(tr["init_lr"])),
            ("Peak LR",              str(tr["peak_lr"])),
            ("Warmup Epochs",        str(tr["warmup_epochs"])),
            ("Max Epochs",           str(tr["max_epochs"])),
            ("Batch Size",           str(tr["batch_size"])),
            ("Early Stop Patience",  str(tr["early_stopping_patience"])),
            ("Sequence Length",      f"{MODEL_CONFIG['sequence_len']} days"),
            ("N Features",           str(MODEL_CONFIG["n_features"])),
            ("Train End Date",       MODEL_CONFIG["train_end_date"]),
        ]
        st.markdown(f"<div class='hp-table'>", unsafe_allow_html=True)
        for k, v in HP:
            st.markdown(f"""<div class='hp-row'>
              <span class='hp-k'>{k}</span>
              <span class='hp-v'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='sec-title' style='margin-top:1.5rem'>Underfitting Fixes — v2 → v4</div>",
                    unsafe_allow_html=True)

        CHANGES = [
            ("l2",              "1e-5",    "5e-6",         "Too aggressive on small folds"),
            ("recurrent_drop",  "0.1",     "0.05",         "Blocking recurrent info flow"),
            ("dropout",         "0.2",     "0.15",         "Reduced forward-pass noise"),
            ("gru1",            "128",     "160",          "More capacity for 30yr data"),
            ("gru2",            "64",      "96",           "Matched wider first layer"),
            ("kernel_init",     "he_norm", "glorot_unif",  "Better for GRU gates"),
            ("batch_size",      "32",      "16",           "Richer gradients on small folds"),
            ("max_epochs",      "100",     "150",          "Fold4 val falling at ep85"),
            ("patience",        "25",      "35",           "Stop cutting off mid-convergence"),
            ("peak_lr",         "1e-3",    "8e-4",         "Wider layers unstable at 1e-3"),
            ("best fold",       "last()",  "min(val_loss)","Fold4 was underfitting"),
        ]

        st.markdown(f"""<div style='background:{NAVY3};border:1px solid {DIM_BLUE};
            border-radius:6px;overflow:hidden;'>
          <div class='chlog-row' style='background:{NAVY4};font-size:.6rem;
               color:{SILVER2};letter-spacing:.1em;text-transform:uppercase;'>
            <span>Param</span><span>Old</span><span>New</span><span>Reason</span>
          </div>""", unsafe_allow_html=True)
        for p, o, n, r in CHANGES:
            st.markdown(f"""<div class='chlog-row'>
              <span class='p'>{p}</span>
              <span class='o'>{o}</span>
              <span class='n'>{n}</span>
              <span class='r'>{r}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""<div class='disc'>
          ⚠ NOT FINANCIAL ADVICE · Research prototype only.<br>
          Past model performance does not guarantee future results.</div>""",
          unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

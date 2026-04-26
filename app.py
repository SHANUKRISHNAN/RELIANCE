"""
RELIANCE Industries — AttentionGRU Forecast Dashboard
Streamlit Cloud compatible (Altair only — ships with Streamlit, no extra installs)
Run locally : streamlit run app.py
Deploy      : push to GitHub → connect on share.streamlit.io
"""

import os, json, warnings
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RELIANCE · GRU Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
DARK_BG      = "#0B0F1A"
CARD_BG      = "#111827"
BORDER       = "#1E293B"
ACCENT_BLUE  = "#3B82F6"
ACCENT_TEAL  = "#14B8A6"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED   = "#EF4444"
ACCENT_GREEN = "#22C55E"
TEXT_PRIMARY = "#F1F5F9"
TEXT_MUTED   = "#64748B"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {{
      background-color:{DARK_BG}; color:{TEXT_PRIMARY}; font-family:'Inter',sans-serif;
  }}

  /* Hero */
  .hero {{ background:linear-gradient(135deg,#0B0F1A 0%,#0F172A 60%,#0B1629 100%);
           border-bottom:1px solid {BORDER}; padding:2rem 2rem 1.6rem;
           margin:-1rem -1rem 2rem -1rem; position:relative; overflow:hidden; }}
  .hero::before {{ content:''; position:absolute; top:-80px; right:-80px;
                   width:320px; height:320px;
                   background:radial-gradient(circle,{ACCENT_BLUE}14 0%,transparent 70%);
                   pointer-events:none; }}
  .hero-tag   {{ font-family:'DM Mono',monospace; font-size:.72rem; color:{ACCENT_TEAL};
                 letter-spacing:.14em; text-transform:uppercase; margin:0 0 .3rem; }}
  .hero-title {{ font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
                 color:{TEXT_PRIMARY}; letter-spacing:-.02em; margin:0; }}
  .badge      {{ display:inline-block; background:{ACCENT_BLUE}22;
                 border:1px solid {ACCENT_BLUE}55; color:{ACCENT_BLUE};
                 font-family:'DM Mono',monospace; font-size:.7rem;
                 padding:.18rem .55rem; border-radius:4px; margin-left:.75rem; vertical-align:middle; }}

  /* Metric cards */
  .mc {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px;
         padding:1.1rem 1.4rem; position:relative; overflow:hidden; margin-bottom:.1rem; }}
  .mc::after {{ content:''; position:absolute; top:0;left:0;right:0; height:2px;
                background:linear-gradient(90deg,{ACCENT_BLUE},{ACCENT_TEAL}); }}
  .mc-lbl {{ font-family:'DM Mono',monospace; font-size:.66rem; color:{TEXT_MUTED};
             letter-spacing:.1em; text-transform:uppercase; margin-bottom:.35rem; }}
  .mc-val {{ font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:700;
             color:{TEXT_PRIMARY}; line-height:1; }}
  .mc-sub {{ font-size:.72rem; color:{TEXT_MUTED}; margin-top:.3rem; }}
  .g {{ color:{ACCENT_GREEN}!important; }} .a {{ color:{ACCENT_AMBER}!important; }}
  .r {{ color:{ACCENT_RED}!important;   }} .t {{ color:{ACCENT_TEAL}!important;  }}

  /* Section headers */
  .sh {{ font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
         color:{TEXT_PRIMARY}; border-left:3px solid {ACCENT_BLUE};
         padding-left:.7rem; margin:1.8rem 0 .8rem; }}

  /* Forecast table */
  .ft {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px; overflow:hidden; }}
  .fr {{ display:flex; align-items:center; padding:.5rem 1rem;
         border-bottom:1px solid {BORDER}; font-size:.82rem; }}
  .fr:last-child {{ border-bottom:none; }}
  .fd {{ font-family:'DM Mono',monospace; color:{TEXT_MUTED}; flex:1; }}
  .fp {{ font-family:'DM Mono',monospace; font-weight:500; flex:1; text-align:right; }}
  .fc {{ flex:1; text-align:right; font-family:'DM Mono',monospace; font-size:.77rem; }}
  .up {{ color:{ACCENT_GREEN}; }} .dn {{ color:{ACCENT_RED}; }}

  /* Hyperparameter table */
  .hp-row {{ display:flex; justify-content:space-between; padding:.42rem 1rem;
             border-bottom:1px solid {BORDER}; font-size:.8rem; }}
  .hp-k {{ color:{TEXT_MUTED}; }} .hp-v {{ font-family:'DM Mono',monospace; color:{ACCENT_TEAL}; }}

  /* Arch layer */
  .al {{ display:flex; align-items:center; margin-bottom:3px; }}
  .al-box {{ border-radius:6px; padding:.35rem .8rem; min-width:110px;
             text-align:center; font-family:'DM Mono',monospace; font-size:.73rem;
             font-weight:500; border:1px solid; }}
  .al-desc {{ margin-left:.7rem; font-size:.73rem; color:{TEXT_MUTED}; }}
  .al-arrow {{ margin-left:52px; color:#334155; font-size:.85rem; line-height:.7; margin-bottom:2px; }}

  /* Pills */
  .pill {{ display:inline-block; background:{BORDER}; border:1px solid #2D3748;
           border-radius:6px; padding:.28rem .65rem; font-family:'DM Mono',monospace;
           font-size:.7rem; color:{TEXT_MUTED}; margin:.2rem; }}

  /* Disclaimer */
  .disc {{ background:{ACCENT_AMBER}11; border:1px solid {ACCENT_AMBER}44;
           border-radius:8px; padding:.7rem 1rem; font-size:.76rem;
           color:{ACCENT_AMBER}; margin-top:1.5rem; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{ background:#0D1321; border-right:1px solid {BORDER}; }}
  .sb-lbl {{ font-family:'DM Mono',monospace; font-size:.68rem; color:{TEXT_MUTED};
             letter-spacing:.1em; text-transform:uppercase; margin-bottom:.4rem; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; border-bottom:1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"]      {{ font-family:'DM Mono',monospace; font-size:.76rem;
                                       letter-spacing:.05em; color:{TEXT_MUTED};
                                       border-radius:6px 6px 0 0; padding:.45rem .9rem; }}
  .stTabs [aria-selected="true"]    {{ color:{TEXT_PRIMARY}!important;
                                       background:{CARD_BG}!important;
                                       border-top:2px solid {ACCENT_BLUE}!important; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  .block-container {{ padding-top:0!important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ALTAIR DARK THEME WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
def dark(chart, h=300):
    return (
        chart.properties(height=h)
        .configure(background="transparent",
                   view=alt.ViewConfig(stroke=BORDER))
        .configure_axis(gridColor=BORDER, domainColor=BORDER,
                        labelColor=TEXT_MUTED, titleColor=TEXT_MUTED,
                        labelFont="DM Mono, monospace", titleFont="DM Mono, monospace",
                        labelFontSize=10, titleFontSize=10)
        .configure_legend(labelColor=TEXT_MUTED, titleColor=TEXT_MUTED,
                          labelFont="DM Mono, monospace", titleFont="DM Mono, monospace",
                          strokeColor=BORDER)
        .configure_title(color=TEXT_MUTED, font="DM Mono, monospace", fontSize=11)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARTIFACT LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(model_dir: str):
    try:
        import tensorflow as tf

        class BahdanauAttention(tf.keras.layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs); self.units = units
                self.W = tf.keras.layers.Dense(units)
                self.V = tf.keras.layers.Dense(1)
            def call(self, h):
                s = self.V(tf.nn.tanh(self.W(h)))
                s -= tf.reduce_max(s, axis=1, keepdims=True)
                a  = tf.nn.softmax(s, axis=1)
                return tf.reduce_sum(a * h, axis=1), a
            def get_config(self):
                c = super().get_config(); c["units"] = self.units; return c

        with open(os.path.join(model_dir, "model_config.json")) as f:
            config = json.load(f)
        model    = tf.keras.models.load_model(
            os.path.join(model_dir, "final_model.keras"),
            custom_objects={"BahdanauAttention": BahdanauAttention})
        f_scaler = joblib.load(os.path.join(model_dir, "final_f_scaler.pkl"))
        t_scaler = joblib.load(os.path.join(model_dir, "final_t_scaler.pkl"))
        return model, f_scaler, t_scaler, config, None
    except Exception as e:
        return None, None, None, None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (mirrors training exactly)
# ══════════════════════════════════════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
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


# ══════════════════════════════════════════════════════════════════════════════
#  FORECAST ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_forecast(model, f_scaler, t_scaler, config, df_feat, n_days):
    seq_len      = config["sequence_len"]
    feature_cols = config["feature_cols"]
    if len(df_feat) < seq_len:
        return None, f"Need ≥{seq_len} rows, got {len(df_feat)}"

    last_rows   = df_feat[feature_cols].iloc[-seq_len:].values
    seq         = f_scaler.transform(last_rows)[np.newaxis, :, :]
    avg_log_vol = float(df_feat["Log_Volume"].iloc[-30:].mean())
    avg_hl      = float(df_feat["HL_Ratio"].iloc[-30:].mean())
    last_close  = float(df_feat["Close"].iloc[-1])
    last_date   = df_feat.index[-1]

    close_buf = list(last_rows[:, feature_cols.index("Close")])
    lr_buf    = []
    rows      = []

    for day in range(n_days):
        ps      = model.predict(seq, verbose=0)
        pred_lr = float(t_scaler.inverse_transform(ps)[0, 0])
        prev_c  = close_buf[-1]
        new_c   = prev_c * np.exp(pred_lr)
        close_buf.append(new_c); lr_buf.append(pred_lr)
        rows.append({"day": day+1, "pred_lr": pred_lr,
                     "new_close": new_c,
                     "pct": (new_c / last_close - 1) * 100,
                     "direction": "UP" if pred_lr >= 0 else "DOWN"})
        cb      = close_buf
        raw_row = np.array([[
            prev_c, new_c*(1+avg_hl/2), new_c*(1-avg_hl/2), new_c,
            np.exp(avg_log_vol), pred_lr, avg_hl, (new_c-prev_c)/prev_c,
            (new_c-cb[-6])  if len(cb)>5  else 0.0,
            (new_c-cb[-11]) if len(cb)>10 else 0.0,
            float(np.std(lr_buf[-9:]+[pred_lr])) if lr_buf else 0.0,
            (np.mean(cb[-5:])/np.mean(cb[-20:])) if len(cb)>=20 else 1.0,
            avg_log_vol,
        ]])
        seq = np.roll(seq, -1, axis=1)
        seq[0,-1,:] = f_scaler.transform(raw_row)[0]

    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    out   = pd.DataFrame(rows); out["Date"] = dates
    return out, None


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""<div style='font-family:Syne,sans-serif;font-size:1.05rem;
        font-weight:700;color:{TEXT_PRIMARY};padding:.4rem 0 1.2rem;
        border-bottom:1px solid {BORDER};margin-bottom:1.2rem;'>⚙️ Configuration</div>""",
        unsafe_allow_html=True)

    st.markdown("<div class='sb-lbl'>Model directory</div>", unsafe_allow_html=True)
    model_dir = st.text_input("_md", value="outputs/models", label_visibility="collapsed")

    st.markdown("<div class='sb-lbl' style='margin-top:.9rem'>Upload stock CSV</div>",
                unsafe_allow_html=True)
    csv_file = st.file_uploader("_csv", type=["csv"], label_visibility="collapsed",
                                 help="Required columns: Date, Open, High, Low, Close, Volume")

    st.markdown("<div class='sb-lbl' style='margin-top:.9rem'>Forecast horizon (days)</div>",
                unsafe_allow_html=True)
    n_days = st.slider("_nd", 1, 30, 10, label_visibility="collapsed")

    st.markdown("<div class='sb-lbl' style='margin-top:.9rem'>Historical window</div>",
                unsafe_allow_html=True)
    hist_days = st.select_slider("_hd", options=[30,60,90,180,365], value=90,
                                  label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{BORDER};margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("<div class='sb-lbl'>Pre-saved forecast CSV</div>", unsafe_allow_html=True)
    forecast_csv_file = st.file_uploader("_fcsv", type=["csv"], key="fcsv",
                                          label_visibility="collapsed")
    st.markdown("<div class='sb-lbl' style='margin-top:.9rem'>Pre-saved metrics CSV</div>",
                unsafe_allow_html=True)
    metrics_csv_file = st.file_uploader("_mcsv", type=["csv"], key="mcsv",
                                         label_visibility="collapsed")

    st.markdown(f"""<div style='margin-top:2rem;font-family:DM Mono,monospace;
        font-size:.63rem;color:#334155;line-height:1.7;'>
        AttentionGRU_v4 · RELIANCE Industries<br>
        Walk-forward cross-validation<br>
        Huber loss · Adam · Bahdanau attention</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='hero'>
  <p class='hero-tag'>NSE · RELIANCE · Attention-GRU Forecasting System</p>
  <h1 class='hero-title'>RELIANCE Stock Forecast
    <span class='badge'>AttentionGRU_v4</span>
  </h1>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
model, f_scaler, t_scaler, config, load_err = load_artifacts(model_dir)

if load_err:
    st.error(f"**Model loading failed:** {load_err}")
    st.code("""Expected structure:\noutputs/models/\n├── final_model.keras\n├── final_f_scaler.pkl\n├── final_t_scaler.pkl\n└── model_config.json""")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD STOCK DATA
# ══════════════════════════════════════════════════════════════════════════════
df_raw = df_feat = None
if csv_file:
    try:
        df_raw = pd.read_csv(csv_file)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True)
        df_raw.set_index("Date", inplace=True); df_raw.sort_index(inplace=True)
        df_raw = df_raw[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")
        df_raw["Volume"] = df_raw["Volume"].replace(0, np.nan); df_raw.dropna(inplace=True)
        df_feat = add_features(df_raw)
    except Exception as e:
        st.error(f"CSV parse error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  TOP METRIC STRIP
# ══════════════════════════════════════════════════════════════════════════════
if config:
    arch = config.get("architecture", {}); tr_cfg = config.get("training", {})
    c1,c2,c3,c4,c5 = st.columns(5)
    def metric_card(col, label, value, sub, cls=""):
        col.markdown(f"""<div class='mc'><div class='mc-lbl'>{label}</div>
          <div class='mc-val {cls}' style='font-size:1.05rem;padding-top:.15rem'>{value}</div>
          <div class='mc-sub'>{sub}</div></div>""", unsafe_allow_html=True)

    metric_card(c1,"Best Fold",       f"#{config.get('trained_on_fold','—')}","Selected by min val_loss")
    metric_card(c2,"Architecture",    f"GRU({arch.get('gru1','?')},{arch.get('gru2','?')})","+ Bahdanau Attention")
    metric_card(c3,"Train End Date",  config.get("train_end_date","—"),"Walk-forward fold cutoff")
    metric_card(c4,"Sequence Length", f"{config.get('sequence_len','—')} days",f"{config.get('n_features','—')} input features")
    metric_card(c5,"Loss / Optimiser",f"{str(tr_cfg.get('loss','?')).title()} / Adam",
                f"LR {tr_cfg.get('peak_lr','?')} · batch {tr_cfg.get('batch_size','?')}")


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4 = st.tabs(["📈  Forecast", "📊  Metrics", "📜  History", "🔬  Architecture"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with t1:
    forecast_df = None; fc_err = None

    if model is not None and df_feat is not None:
        with st.spinner("Running inference…"):
            forecast_df, fc_err = run_forecast(model, f_scaler, t_scaler, config, df_feat, n_days)

    elif forecast_csv_file:
        try:
            raw = pd.read_csv(forecast_csv_file, parse_dates=["Date"])
            forecast_df = pd.DataFrame({
                "Date":      pd.to_datetime(raw["Date"]),
                "new_close": raw.get("Predicted_Close", raw.get("new_close", 0)),
                "pct":       raw.get("Pct_vs_last",     raw.get("pct", 0)),
                "pred_lr":   0.0,
                "day":       range(1, len(raw)+1),
            })
            forecast_df["direction"] = forecast_df["pct"].apply(lambda x: "UP" if x >= 0 else "DOWN")
            st.info("Showing pre-loaded forecast CSV. Upload a stock CSV for live inference.")
        except Exception as e:
            fc_err = str(e)

    if fc_err:
        st.error(f"Forecast error: {fc_err}")

    if forecast_df is not None and not forecast_df.empty:
        last_close = float(df_feat["Close"].iloc[-1]) if df_feat is not None \
                     else float(forecast_df["new_close"].iloc[0])
        final_pred = float(forecast_df["new_close"].iloc[-1])
        final_pct  = float(forecast_df["pct"].iloc[-1])
        days_up    = int((forecast_df["direction"] == "UP").sum())
        days_dn    = len(forecast_df) - days_up

        # Summary cards
        m1,m2,m3,m4 = st.columns(4)
        last_date_str = df_feat.index[-1].strftime("%d %b %Y") if df_feat is not None else "—"
        metric_card(m1,"Last Close",f"₹{last_close:,.2f}",last_date_str)
        pct_cls = "g" if final_pct >= 0 else "r"
        arrow   = "▲" if final_pct >= 0 else "▼"
        m2.markdown(f"""<div class='mc'><div class='mc-lbl'>Day {len(forecast_df)} Target</div>
          <div class='mc-val' style='font-size:1.05rem;padding-top:.15rem'>₹{final_pred:,.2f}</div>
          <div class='mc-sub {pct_cls}'>{arrow} {abs(final_pct):.2f}% vs today</div></div>""",
          unsafe_allow_html=True)
        metric_card(m3,"Bullish Days",  f"<span class='g'>{days_up}</span>",
                    f"out of {len(forecast_df)} forecast days")
        metric_card(m4,"Bearish Days",  f"<span class='r'>{days_dn}</span>",
                    f"out of {len(forecast_df)} forecast days")

        st.markdown("")
        fc_left, fc_right = st.columns([2, 1])

        with fc_left:
            st.markdown("<div class='sh'>Price Forecast Trajectory</div>", unsafe_allow_html=True)

            chart_rows = []
            if df_raw is not None:
                tail = df_raw["Close"].iloc[-hist_days:].reset_index()
                tail.columns = ["Date","Price"]
                tail["Type"] = "Historical"; tail["Dir"] = "—"
                chart_rows.append(tail)

            fc_plot = forecast_df[["Date","new_close","direction"]].copy()
            fc_plot.columns = ["Date","Price","Dir"]
            fc_plot["Type"] = "Forecast"
            chart_rows.append(fc_plot)

            plot_df = pd.concat(chart_rows, ignore_index=True)
            plot_df["Date"] = pd.to_datetime(plot_df["Date"])

            hist_line = alt.Chart(
                plot_df[plot_df["Type"]=="Historical"]
            ).mark_line(color=ACCENT_BLUE, strokeWidth=1.8).encode(
                x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", title="")),
                y=alt.Y("Price:Q", axis=alt.Axis(title="Price (₹)", format=",.0f")),
                tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                         alt.Tooltip("Price:Q", format=",.2f", title="Close ₹")]
            )
            fc_line = alt.Chart(
                plot_df[plot_df["Type"]=="Forecast"]
            ).mark_line(color=ACCENT_TEAL, strokeWidth=2, strokeDash=[4,2]).encode(
                x="Date:T", y="Price:Q"
            )
            fc_dots = alt.Chart(
                plot_df[plot_df["Type"]=="Forecast"]
            ).mark_circle(size=65).encode(
                x="Date:T", y="Price:Q",
                color=alt.Color("Dir:N",
                    scale=alt.Scale(domain=["UP","DOWN"],
                                    range=[ACCENT_GREEN, ACCENT_RED]),
                    legend=alt.Legend(title="Direction")),
                tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                         alt.Tooltip("Price:Q", format=",.2f", title="₹"), "Dir:N"]
            )

            st.altair_chart(dark(hist_line + fc_line + fc_dots, h=340), use_container_width=True)

            # Log-return bar chart
            if "pred_lr" in forecast_df.columns and forecast_df["pred_lr"].abs().sum() > 0:
                st.markdown("<div class='sh'>Predicted Daily Log Returns</div>", unsafe_allow_html=True)
                lr_df = forecast_df[["Date","pred_lr","direction"]].copy()
                lr_df["lr_pct"] = lr_df["pred_lr"] * 100
                bars = alt.Chart(lr_df).mark_bar(
                    cornerRadiusTopLeft=3, cornerRadiusTopRight=3
                ).encode(
                    x=alt.X("Date:T", axis=alt.Axis(format="%d %b", title="")),
                    y=alt.Y("lr_pct:Q", axis=alt.Axis(title="Log return (%)")),
                    color=alt.Color("direction:N",
                        scale=alt.Scale(domain=["UP","DOWN"],
                                        range=[ACCENT_GREEN, ACCENT_RED]), legend=None),
                    tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                             alt.Tooltip("lr_pct:Q", format=".4f", title="Log return %"),
                             "direction:N"]
                )
                zero_rule = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(
                    color=TEXT_MUTED, strokeDash=[3,3], opacity=0.5
                ).encode(y="y:Q")
                st.altair_chart(dark(bars + zero_rule, h=180), use_container_width=True)

        with fc_right:
            st.markdown("<div class='sh'>Day-by-Day Table</div>", unsafe_allow_html=True)
            st.markdown("<div class='ft'>", unsafe_allow_html=True)
            st.markdown(f"""<div class='fr' style='opacity:.45;font-size:.67rem;
                letter-spacing:.08em;text-transform:uppercase;border-bottom:1px solid {BORDER}'>
                <span class='fd'>Date</span>
                <span class='fp'>Price ₹</span>
                <span class='fc'>Chg%</span></div>""", unsafe_allow_html=True)
            for _, row in forecast_df.iterrows():
                d    = pd.to_datetime(row["Date"]).strftime("%d %b")
                p    = f"{float(row['new_close']):,.2f}"
                pv   = float(row["pct"])
                ps   = f"+{pv:.2f}%" if pv >= 0 else f"{pv:.2f}%"
                dcls = "up" if row["direction"] == "UP" else "dn"
                arr  = "▲" if row["direction"] == "UP" else "▼"
                st.markdown(f"""<div class='fr'>
                    <span class='fd'>{d}</span>
                    <span class='fp'>{p}</span>
                    <span class='fc {dcls}'>{arr} {ps}</span></div>""",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(f"""<div style='background:{CARD_BG};border:1px solid {BORDER};
            border-radius:12px;padding:3rem;text-align:center;margin:1rem 0;'>
          <div style='font-size:2.5rem;margin-bottom:1rem'>📂</div>
          <div style='font-family:Syne,sans-serif;font-size:1.05rem;color:#94A3B8;margin-bottom:.5rem;'>
            Upload a stock CSV to run live inference</div>
          <div style='font-family:DM Mono,monospace;font-size:.73rem;color:#475569;'>
            Required columns: Date · Open · High · Low · Close · Volume</div>
          <div style='font-family:DM Mono,monospace;font-size:.7rem;color:#334155;margin-top:.75rem;'>
            — or load a pre-saved forecast CSV in the sidebar —</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — METRICS
# ─────────────────────────────────────────────────────────────────────────────
with t2:
    if metrics_csv_file:
        try:
            metrics_df = pd.read_csv(metrics_csv_file)
        except Exception:
            metrics_df = None
    else:
        metrics_df = None

    if metrics_df is None:
        metrics_df = pd.DataFrame([
            {"Fold":"1 ★","MAPE_pct":4.942,"R2":0.9560,"DirAcc":54.43},
            {"Fold":"2",  "MAPE_pct":6.040,"R2":0.8595,"DirAcc":51.05},
            {"Fold":"3",  "MAPE_pct":3.742,"R2":0.8028,"DirAcc":50.11},
            {"Fold":"4",  "MAPE_pct":4.850,"R2":0.9531,"DirAcc":52.11},
            {"Fold":"TEST","MAPE_pct":3.164,"R2":0.8799,"DirAcc":50.45},
        ])

    test_row  = metrics_df[metrics_df["Fold"].astype(str).str.upper().str.contains("TEST")]
    fold_rows = metrics_df[~metrics_df["Fold"].astype(str).str.upper().str.contains("TEST")].copy()

    if not test_row.empty:
        tr_r = test_row.iloc[0]
        mape_v = float(tr_r["MAPE_pct"]); r2_v = float(tr_r["R2"]); da_v = float(tr_r["DirAcc"])
        st.markdown("<div class='sh'>Hold-out Test Set Performance</div>", unsafe_allow_html=True)
        ma,mb,mc_,md = st.columns(4)
        metric_card(ma,"Test MAPE",    f"<span class='{'g' if mape_v<5 else 'a'}'>{mape_v:.3f}%</span>","Mean Absolute % Error")
        metric_card(mb,"Test R²",      f"<span class='{'g' if r2_v>.85 else 'a'}'>{r2_v:.4f}</span>","Variance explained")
        metric_card(mc_,"Dir. Accuracy",f"<span class='{'g' if da_v>53 else 'a'}'>{da_v:.2f}%</span>","vs 50% random baseline")
        edge = da_v - 50.0
        metric_card(md,"Edge Over Random",f"<span class='{'g' if edge>3 else 'a'}'>+{edge:.2f}%</span>","Directional edge")

    st.markdown("<div class='sh'>Walk-Forward Fold Comparison</div>", unsafe_allow_html=True)

    fold_plot = fold_rows.copy()
    fold_plot["Fold"]  = fold_plot["Fold"].astype(str)
    fold_plot["Color"] = fold_plot["Fold"].apply(lambda x: ACCENT_TEAL if "★" in x else ACCENT_BLUE)

    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        b = alt.Chart(fold_plot).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y("MAPE_pct:Q", axis=alt.Axis(title="MAPE %")),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip("MAPE_pct:Q", format=".3f", title="MAPE %")]
        ).properties(title="MAPE %")
        st.altair_chart(dark(b, h=220), use_container_width=True)

    with ch2:
        b2 = alt.Chart(fold_plot).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y("R2:Q", axis=alt.Axis(title="R²"), scale=alt.Scale(domain=[0,1])),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip("R2:Q", format=".4f", title="R²")]
        ).properties(title="R² Score")
        st.altair_chart(dark(b2, h=220), use_container_width=True)

    with ch3:
        rule50 = alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(
            color=ACCENT_AMBER, strokeDash=[4,3], opacity=0.7
        ).encode(y="y:Q")
        b3 = alt.Chart(fold_plot).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Fold:N", axis=alt.Axis(title="")),
            y=alt.Y("DirAcc:Q", axis=alt.Axis(title="Dir. Acc %"),
                    scale=alt.Scale(domain=[45,60])),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=["Fold:N", alt.Tooltip("DirAcc:Q", format=".2f", title="Dir. Acc %")]
        ).properties(title="Directional Accuracy %")
        st.altair_chart(dark(b3 + rule50, h=220), use_container_width=True)

    st.markdown("<div class='sh'>Full Metrics Table</div>", unsafe_allow_html=True)
    disp = metrics_df[["Fold","MAPE_pct","R2","DirAcc"]].copy()
    disp.columns = ["Fold","MAPE %","R²","Dir. Acc %"]
    st.dataframe(disp.style.format({"MAPE %":"{:.3f}","R²":"{:.4f}","Dir. Acc %":"{:.2f}"}),
                 use_container_width=True, height=210)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with t3:
    if df_raw is not None and not df_raw.empty:
        hist = df_raw.iloc[-hist_days:].copy().reset_index()
        hist.columns = ["Date"] + list(hist.columns[1:])
        hist["Date"]       = pd.to_datetime(hist["Date"])
        hist["MA20"]       = hist["Close"].rolling(20).mean()
        hist["MA5"]        = hist["Close"].rolling(5).mean()
        hist["Dir"]        = (hist["Close"] >= hist["Open"]).map({True:"UP",False:"DOWN"})
        hist["Return_pct"] = hist["Close"].pct_change() * 100

        st.markdown("<div class='sh'>Closing Price with Moving Averages</div>", unsafe_allow_html=True)
        cl = alt.Chart(hist).mark_line(color=ACCENT_BLUE, strokeWidth=1.6).encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", title="")),
            y=alt.Y("Close:Q", axis=alt.Axis(title="Price (₹)", format=",.0f")),
            tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                     alt.Tooltip("Close:Q", format=",.2f", title="Close ₹")]
        )
        m5  = alt.Chart(hist.dropna(subset=["MA5"])).mark_line(
            color=ACCENT_TEAL, strokeWidth=1, strokeDash=[3,2], opacity=0.8
        ).encode(x="Date:T", y=alt.Y("MA5:Q", title=""))
        m20 = alt.Chart(hist.dropna(subset=["MA20"])).mark_line(
            color=ACCENT_AMBER, strokeWidth=1, strokeDash=[5,3], opacity=0.8
        ).encode(x="Date:T", y=alt.Y("MA20:Q", title=""))
        st.altair_chart(dark(cl + m5 + m20, h=320), use_container_width=True)

        h1, h2 = st.columns(2)
        with h1:
            st.markdown("<div class='sh'>Volume</div>", unsafe_allow_html=True)
            vb = alt.Chart(hist).mark_bar(opacity=0.78).encode(
                x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", title="")),
                y=alt.Y("Volume:Q", axis=alt.Axis(title="Volume")),
                color=alt.Color("Dir:N",
                    scale=alt.Scale(domain=["UP","DOWN"],
                                    range=[ACCENT_GREEN,ACCENT_RED]), legend=None),
                tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"),
                         alt.Tooltip("Volume:Q", format=",.0f")]
            )
            st.altair_chart(dark(vb, h=200), use_container_width=True)

        with h2:
            st.markdown("<div class='sh'>Daily Return Distribution</div>", unsafe_allow_html=True)
            rh = alt.Chart(hist.dropna(subset=["Return_pct"])).mark_bar(
                color=ACCENT_BLUE, opacity=0.8
            ).encode(
                x=alt.X("Return_pct:Q", bin=alt.Bin(maxbins=55),
                         axis=alt.Axis(title="Daily Return (%)")),
                y=alt.Y("count()", axis=alt.Axis(title="Count")),
                tooltip=[alt.Tooltip("Return_pct:Q", bin=True, title="Return %"), "count()"]
            )
            mean_v = hist["Return_pct"].mean()
            mr = alt.Chart(pd.DataFrame({"x":[mean_v]})).mark_rule(
                color=ACCENT_TEAL, strokeDash=[4,3]
            ).encode(x="x:Q")
            st.altair_chart(dark(rh + mr, h=200), use_container_width=True)

        st.markdown("<div class='sh'>Summary Statistics</div>", unsafe_allow_html=True)
        st.dataframe(
            hist[["Open","High","Low","Close","Volume"]].describe().T.style.format("{:.2f}"),
            use_container_width=True)
    else:
        st.markdown(f"""<div style='background:{CARD_BG};border:1px solid {BORDER};
            border-radius:12px;padding:3rem;text-align:center;'>
          <div style='font-size:2rem;margin-bottom:.75rem'>📉</div>
          <div style='font-family:Syne,sans-serif;font-size:1rem;color:#94A3B8;'>
            Upload a stock CSV in the sidebar to view price history</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
with t4:
    if config:
        arch   = config.get("architecture", {})
        tr_cfg = config.get("training", {})
        feat   = config.get("feature_cols", [])

        a1, a2 = st.columns(2)

        with a1:
            st.markdown("<div class='sh'>Model Architecture</div>", unsafe_allow_html=True)
            layers_info = [
                ("INPUT",     f"(60 steps × {config.get('n_features',13)} features)", ACCENT_BLUE),
                ("GRU 1",     f"{arch.get('gru1','?')} units · return_sequences=True", ACCENT_TEAL),
                ("LayerNorm", "Normalise across the feature dimension", TEXT_MUTED),
                ("GRU 2",     f"{arch.get('gru2','?')} units · return_sequences=True", ACCENT_TEAL),
                ("LayerNorm", "Normalise across the feature dimension", TEXT_MUTED),
                ("Attention", f"Bahdanau additive · {arch.get('attn_units','?')} units", ACCENT_AMBER),
                ("Dense 1",   f"{arch.get('dense1','?')} units · ReLU · Dropout {arch.get('dropout','?')}", ACCENT_BLUE),
                ("Dense 2",   f"{arch.get('dense2','?')} units · ReLU · Dropout {float(arch.get('dropout',0.15))*0.5:.3f}", ACCENT_BLUE),
                ("OUTPUT",    "1 unit → predicted Log-Return", ACCENT_GREEN),
            ]
            for i,(name,desc,color) in enumerate(layers_info):
                st.markdown(f"""
                <div class='al'>
                  <div class='al-box' style='color:{color};border-color:{color}55;background:{color}18;'>{name}</div>
                  <div class='al-desc'>{desc}</div>
                </div>
                {"<div class='al-arrow'>↓</div>" if i < len(layers_info)-1 else ""}
                """, unsafe_allow_html=True)

        with a2:
            st.markdown("<div class='sh'>Hyperparameters</div>", unsafe_allow_html=True)
            hp = [
                ("L2 Regularisation",   arch.get("l2","?")),
                ("Recurrent Dropout",   arch.get("recurrent_dropout","?")),
                ("Dropout",             arch.get("dropout","?")),
                ("Batch Size",          tr_cfg.get("batch_size","?")),
                ("Max Epochs",          tr_cfg.get("max_epochs","?")),
                ("Early Stop Patience", tr_cfg.get("early_stopping_patience","?")),
                ("Init LR",             tr_cfg.get("init_lr","?")),
                ("Peak LR",             tr_cfg.get("peak_lr","?")),
                ("Warmup Epochs",       tr_cfg.get("warmup_epochs","?")),
                ("Loss Function",       str(tr_cfg.get("loss","?")).title()),
                ("Kernel Init",         "glorot_uniform"),
                ("Sequence Length",     f"{config.get('sequence_len','?')} days"),
            ]
            st.markdown(f"<div style='background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;overflow:hidden;'>",
                        unsafe_allow_html=True)
            for k,v in hp:
                st.markdown(f"""<div class='hp-row'>
                  <span class='hp-k'>{k}</span>
                  <span class='hp-v'>{v}</span></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='sh'>Input Features</div>", unsafe_allow_html=True)
        pills = "".join([f"<span class='pill'>{f}</span>" for f in feat])
        st.markdown(f"<div style='margin-bottom:.5rem'>{pills}</div>", unsafe_allow_html=True)
        st.markdown(f"""<div style='font-family:DM Mono,monospace;font-size:.7rem;color:{TEXT_MUTED};'>
          Target: <span style='color:{ACCENT_AMBER}'>Log_Return</span>
          &nbsp;·&nbsp; Features → RobustScaler (fit on train only)
          &nbsp;·&nbsp; Target → StandardScaler (fit on train only)
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class='disc'>
      ⚠️ <strong>Not financial advice.</strong> Research prototype only.
      Forecast reliability degrades rapidly beyond 5–7 days.
      Past model performance does not guarantee future results.</div>""",
      unsafe_allow_html=True)

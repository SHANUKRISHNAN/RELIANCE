"""
RELIANCE Industries — AttentionGRU Forecast Terminal
Streamlit Cloud compatible · Altair only · no Plotly
"""

import os, json, warnings, base64
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RELIANCE · GRU Terminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BG0   = "#060810"
BG1   = "#0A0D18"
BG2   = "#0F1425"
BG3   = "#141929"
GRID  = "#1A2035"
LINE  = "#222C44"
CYAN  = "#00E5FF"
LIME  = "#39FF14"
RED   = "#FF2D55"
GOLD  = "#FFD166"
MUTED = "#4A5568"
DIM   = "#8892A4"
BRIGHT= "#E8EDF5"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Exo+2:wght@300;400;600;700;900&family=JetBrains+Mono:wght@300;400;500&display=swap');
html,body,[class*="css"]{{background:{BG0};color:{BRIGHT};font-family:'JetBrains Mono',monospace;}}
.block-container{{padding:0!important;max-width:100%!important;}}
#MainMenu,footer,header{{visibility:hidden;}}
[data-testid="stSidebar"]{{display:none;}}
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{BG0};}}
::-webkit-scrollbar-thumb{{background:{LINE};border-radius:2px;}}

.topbar{{background:{BG1};border-bottom:1px solid {LINE};padding:.7rem 1.5rem;
  display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;}}
.topbar-brand{{font-family:'Exo 2',sans-serif;font-weight:900;font-size:1.05rem;
  color:{CYAN};letter-spacing:.15em;text-transform:uppercase;display:flex;align-items:center;gap:.5rem;}}
.dot{{width:7px;height:7px;background:{LIME};border-radius:50%;animation:pulse 2s infinite;}}
@keyframes pulse{{0%,100%{{opacity:1;transform:scale(1);}}50%{{opacity:.4;transform:scale(.7);}}}}
.tbadge{{font-size:.63rem;color:{MUTED};letter-spacing:.07em;border:1px solid {LINE};padding:.18rem .55rem;border-radius:3px;}}
.tstat{{font-size:.68rem;}}
.tstat .l{{color:{MUTED};}}
.tstat .v{{color:{BRIGHT};font-weight:700;}}
.tstat .v.up{{color:{LIME};}}
.tstat .v.dn{{color:{RED};}}

.sec-title{{font-family:'Exo 2',sans-serif;font-size:.85rem;font-weight:700;
  color:{BRIGHT};letter-spacing:.08em;text-transform:uppercase;
  margin:1rem 0 .75rem;display:flex;align-items:center;gap:.5rem;}}
.sec-title::after{{content:'';flex:1;height:1px;background:linear-gradient(90deg,{LINE},transparent);}}

.metric-card{{background:{BG2};border:1px solid {LINE};border-radius:4px;
  padding:.85rem 1rem;position:relative;overflow:hidden;}}
.metric-card::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,{CYAN}00,{CYAN}55,{CYAN}00);}}
.metric-card .lbl{{font-size:.57rem;color:{MUTED};letter-spacing:.12em;text-transform:uppercase;margin-bottom:.28rem;}}
.metric-card .val{{font-family:'Exo 2',sans-serif;font-size:1.4rem;font-weight:700;line-height:1;}}
.metric-card .val.cyan{{color:{CYAN};}} .metric-card .val.lime{{color:{LIME};}}
.metric-card .val.red{{color:{RED};}}   .metric-card .val.gold{{color:{GOLD};}}
.metric-card .sub{{font-size:.6rem;color:{DIM};margin-top:.22rem;}}

.fc-table{{background:{BG2};border:1px solid {LINE};border-radius:4px;overflow:hidden;}}
.fc-row{{display:grid;grid-template-columns:1fr 1fr 1fr;padding:.4rem .7rem;
  border-bottom:1px solid {GRID};font-size:.7rem;}}
.fc-row:hover{{background:{BG3};}}
.fc-row:last-child{{border-bottom:none;}}
.fc-row.hdr{{font-size:.57rem;color:{MUTED};letter-spacing:.1em;text-transform:uppercase;background:{BG1};}}
.fc-row .d{{color:{DIM};font-family:'Space Mono',monospace;}}
.fc-row .p{{text-align:right;color:{BRIGHT};font-family:'Space Mono',monospace;}}
.fc-row .c{{text-align:right;font-family:'Space Mono',monospace;font-size:.67rem;}}
.fc-row .c.up{{color:{LIME};}} .fc-row .c.dn{{color:{RED};}}

.plot-frame{{background:{BG2};border:1px solid {LINE};border-radius:4px;
  padding:.4rem;overflow:hidden;margin-bottom:.85rem;}}
.plot-cap{{font-size:.57rem;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;
  margin-top:.35rem;text-align:center;}}

.arch-layer{{display:flex;align-items:center;gap:.7rem;margin-bottom:3px;}}
.arch-box{{font-family:'Space Mono',monospace;font-size:.63rem;font-weight:700;
  padding:.28rem .65rem;border-radius:3px;min-width:95px;text-align:center;border:1px solid;}}
.arch-desc{{font-size:.63rem;color:{DIM};}}
.arch-arrow{{color:{MUTED};margin-left:103px;font-size:.8rem;line-height:.55;margin-bottom:3px;}}

.hp-row{{display:flex;justify-content:space-between;align-items:center;
  padding:.36rem .75rem;border-bottom:1px solid {GRID};font-size:.7rem;}}
.hp-row:last-child{{border-bottom:none;}}
.hp-k{{color:{DIM};}} .hp-v{{font-family:'Space Mono',monospace;color:{CYAN};}}

.pill{{display:inline-block;background:{BG3};border:1px solid {LINE};border-radius:2px;
  padding:.18rem .45rem;font-family:'Space Mono',monospace;font-size:.6rem;color:{DIM};margin:.12rem;}}

.disc{{background:{GOLD}0D;border:1px solid {GOLD}33;border-radius:3px;
  padding:.55rem .85rem;font-size:.63rem;color:{GOLD};margin-top:.85rem;line-height:1.6;}}

.upload-hint{{background:{BG2};border:1px dashed {LINE};border-radius:4px;
  padding:2.5rem 1rem;text-align:center;}}
.upload-hint .icon{{font-size:2rem;margin-bottom:.5rem;}}
.upload-hint .msg{{font-size:.72rem;color:{DIM};}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def img_b64(path):
    with open(path,"rb") as f: return base64.b64encode(f.read()).decode()

def plot_img(path, cap=""):
    if os.path.exists(path):
        b = img_b64(path)
        st.markdown(f"""<div class='plot-frame'>
          <img src='data:image/png;base64,{b}' style='width:100%;border-radius:2px;display:block;'>
          {"<div class='plot-cap'>"+cap+"</div>" if cap else ""}
        </div>""", unsafe_allow_html=True)

def dark_alt(chart, h=280):
    return (chart.properties(height=h)
        .configure(background="transparent",view=alt.ViewConfig(stroke=GRID))
        .configure_axis(gridColor=GRID,domainColor=LINE,labelColor=MUTED,titleColor=DIM,
            labelFont="JetBrains Mono,monospace",titleFont="JetBrains Mono,monospace",
            labelFontSize=10,titleFontSize=10)
        .configure_legend(labelColor=DIM,titleColor=MUTED,
            labelFont="JetBrains Mono,monospace",titleFont="JetBrains Mono,monospace",
            strokeColor=LINE)
        .configure_title(color=DIM,font="JetBrains Mono,monospace",fontSize=10))

PLOT_DIR  = "outputs/plots"
MODEL_DIR = "outputs/models"


# ── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising model…")
def load_model(d):
    try:
        import tensorflow as tf
        class BahdanauAttention(tf.keras.layers.Layer):
            def __init__(self,units,**kw):
                super().__init__(**kw); self.units=units
                self.W=tf.keras.layers.Dense(units); self.V=tf.keras.layers.Dense(1)
            def call(self,h):
                s=self.V(tf.nn.tanh(self.W(h))); s-=tf.reduce_max(s,axis=1,keepdims=True)
                a=tf.nn.softmax(s,axis=1); return tf.reduce_sum(a*h,axis=1),a
            def get_config(self):
                c=super().get_config(); c["units"]=self.units; return c
        with open(os.path.join(d,"model_config.json")) as f: cfg=json.load(f)
        mdl=tf.keras.models.load_model(os.path.join(d,"final_model.keras"),
            custom_objects={"BahdanauAttention":BahdanauAttention})
        fsc=joblib.load(os.path.join(d,"final_f_scaler.pkl"))
        tsc=joblib.load(os.path.join(d,"final_t_scaler.pkl"))
        return mdl,fsc,tsc,cfg,None
    except Exception as e: return None,None,None,None,str(e)


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer(df):
    df=df.copy()
    df["Return"]=df["Close"].pct_change()
    df["Log_Return"]=np.log(df["Close"]/df["Close"].shift(1))
    df["HL_Ratio"]=(df["High"]-df["Low"])/df["Close"]
    df["OC_Ratio"]=(df["Close"]-df["Open"])/df["Open"]
    df["Momentum_5"]=df["Close"]-df["Close"].shift(5)
    df["Momentum_10"]=df["Close"]-df["Close"].shift(10)
    df["Vol_10"]=df["Return"].rolling(10).std()
    df["MA5"]=df["Close"].rolling(5).mean(); df["MA20"]=df["Close"].rolling(20).mean()
    df["MA_Ratio"]=df["MA5"]/df["MA20"]; df.drop(columns=["MA5","MA20"],inplace=True)
    df["Log_Volume"]=np.log(df["Volume"].clip(lower=1))
    df.replace([np.inf,-np.inf],np.nan,inplace=True); df.dropna(inplace=True)
    return df


# ── Forecast engine ───────────────────────────────────────────────────────────
def run_fc(mdl,fsc,tsc,cfg,df_feat,n):
    sl=cfg["sequence_len"]; fc=cfg["feature_cols"]
    if len(df_feat)<sl: return None,f"Need >={sl} rows"
    last=df_feat[fc].iloc[-sl:].values; seq=fsc.transform(last)[np.newaxis,:,:]
    alv=float(df_feat["Log_Volume"].iloc[-30:].mean())
    ahl=float(df_feat["HL_Ratio"].iloc[-30:].mean())
    lc=float(df_feat["Close"].iloc[-1]); ld=df_feat.index[-1]
    cb=list(last[:,fc.index("Close")]); lb=[]; rows=[]
    for d in range(n):
        ps=mdl.predict(seq,verbose=0); plr=float(tsc.inverse_transform(ps)[0,0])
        pc=cb[-1]; nc=pc*np.exp(plr); cb.append(nc); lb.append(plr)
        rows.append({"day":d+1,"pred_lr":plr,"close":nc,"pct":(nc/lc-1)*100,
                     "dir":"UP" if plr>=0 else "DN"})
        c=cb
        rr=np.array([[pc,nc*(1+ahl/2),nc*(1-ahl/2),nc,np.exp(alv),plr,ahl,(nc-pc)/pc,
            (nc-c[-6]) if len(c)>5 else 0.,(nc-c[-11]) if len(c)>10 else 0.,
            float(np.std(lb[-9:]+[plr])) if lb else 0.,
            (np.mean(c[-5:])/np.mean(c[-20:])) if len(c)>=20 else 1.,alv]])
        seq=np.roll(seq,-1,axis=1); seq[0,-1,:]=fsc.transform(rr)[0]
    dates=pd.bdate_range(start=ld+pd.Timedelta(days=1),periods=n)
    out=pd.DataFrame(rows); out["Date"]=dates; return out,None


# ── Session state tab ─────────────────────────────────────────────────────────
if "tab" not in st.session_state: st.session_state.tab="forecast"

mdl,fsc,tsc,cfg,lerr=load_model(MODEL_DIR)
arch=cfg.get("architecture",{}) if cfg else {}
tr  =cfg.get("training",{})     if cfg else {}

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='topbar'>
  <div class='topbar-brand'><span class='dot'></span>⬡ RELIANCE TERMINAL</div>
  <div class='tbadge'>AttentionGRU_v4 · Fold #{cfg.get('trained_on_fold','—') if cfg else '—'}</div>
  <div class='tstat'><span class='l'>SEQ </span><span class='v'>{cfg.get('sequence_len','—') if cfg else '—'}d</span></div>
  <div class='tstat'><span class='l'>FEAT </span><span class='v'>{cfg.get('n_features','—') if cfg else '—'}</span></div>
  <div class='tstat'><span class='l'>LOSS </span><span class='v'>{str(tr.get('loss','—')).upper()}</span></div>
  <div class='tstat'><span class='l'>MAPE </span><span class='v up'>3.164%</span></div>
  <div class='tstat'><span class='l'>R² </span><span class='v up'>0.8799</span></div>
  <div class='tstat'><span class='l'>DIR ACC </span><span class='v'>50.45%</span></div>
  <div style='margin-left:auto;font-size:.63rem;'>
    {'<span style="color:#39FF14;">● LIVE</span>' if not lerr else '<span style="color:#FF2D55;">● MODEL ERR</span>'}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tab buttons ───────────────────────────────────────────────────────────────
TABS=[("forecast","01 / FORECAST"),("plots","02 / ANALYSIS"),
      ("metrics","03 / METRICS"),("arch","04 / ARCHITECTURE")]
tc=st.columns(len(TABS)+5)
for i,(key,label) in enumerate(TABS):
    active=st.session_state.tab==key
    if tc[i].button(label,key=f"t_{key}",
        help=label,
        type="primary" if active else "secondary"):
        st.session_state.tab=key; st.rerun()

st.markdown(f"<hr style='margin:0 0 1rem;border-color:{LINE};'>",unsafe_allow_html=True)
TAB=st.session_state.tab

# ── Settings expander ─────────────────────────────────────────────────────────
with st.expander("⚙  INPUTS & SETTINGS", expanded=(TAB=="forecast")):
    ic1,ic2,ic3,ic4=st.columns([2,1,1,2])
    with ic1:
        st.markdown(f"<div style='font-size:.63rem;color:{MUTED};margin-bottom:.25rem;'>STOCK CSV</div>",unsafe_allow_html=True)
        csv_file=st.file_uploader("csv",type=["csv"],label_visibility="collapsed",key="mcsv")
    with ic2:
        st.markdown(f"<div style='font-size:.63rem;color:{MUTED};margin-bottom:.25rem;'>FORECAST DAYS</div>",unsafe_allow_html=True)
        n_days=st.slider("nd",1,30,10,label_visibility="collapsed")
    with ic3:
        st.markdown(f"<div style='font-size:.63rem;color:{MUTED};margin-bottom:.25rem;'>HISTORY WINDOW</div>",unsafe_allow_html=True)
        hist_days=st.select_slider("hd",[30,60,90,180,365],value=90,label_visibility="collapsed")
    with ic4:
        st.markdown(f"<div style='font-size:.63rem;color:{MUTED};margin-bottom:.25rem;'>PRE-SAVED FORECAST CSV</div>",unsafe_allow_html=True)
        saved_fc=st.file_uploader("sfc",type=["csv"],label_visibility="collapsed",key="sfcsv")

# ── Load stock data ───────────────────────────────────────────────────────────
df_raw=df_feat=None
if csv_file:
    try:
        df_raw=pd.read_csv(csv_file)
        df_raw["Date"]=pd.to_datetime(df_raw["Date"],dayfirst=True)
        df_raw.set_index("Date",inplace=True); df_raw.sort_index(inplace=True)
        cols=[c for c in ["Open","High","Low","Close","Volume"] if c in df_raw.columns]
        df_raw=df_raw[cols].apply(pd.to_numeric,errors="coerce")
        df_raw["Volume"]=df_raw["Volume"].replace(0,np.nan); df_raw.dropna(inplace=True)
        df_feat=engineer(df_raw)
    except Exception as e: st.error(f"CSV error: {e}")

# Fallback to bundled data
if df_raw is None:
    for p in ["outputs/data/RELIANCE.csv","data/RELIANCE.csv","RELIANCE.csv"]:
        if os.path.exists(p):
            try:
                df_raw=pd.read_csv(p); df_raw["Date"]=pd.to_datetime(df_raw["Date"],dayfirst=True)
                df_raw.set_index("Date",inplace=True); df_raw.sort_index(inplace=True)
                df_raw=df_raw[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric,errors="coerce")
                df_raw["Volume"]=df_raw["Volume"].replace(0,np.nan); df_raw.dropna(inplace=True)
                df_feat=engineer(df_raw); break
            except Exception: pass


# ══════════════════════════════════════════════════════════════════
#  TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════════
if TAB=="forecast":
    fc_df=None; fc_err=None

    if mdl is not None and df_feat is not None:
        with st.spinner("Running AttentionGRU inference…"):
            fc_df,fc_err=run_fc(mdl,fsc,tsc,cfg,df_feat,n_days)
    elif saved_fc:
        try:
            raw=pd.read_csv(saved_fc,parse_dates=["Date"])
            fc_df=pd.DataFrame({"Date":pd.to_datetime(raw["Date"]),
                "close":raw.get("Predicted_Close",raw.iloc[:,1]),
                "pct":raw.get("Pct_vs_last",raw.get("pct",0)),
                "pred_lr":0.,"day":range(1,len(raw)+1)})
            fc_df["dir"]=fc_df["pct"].apply(lambda x:"UP" if float(x)>=0 else "DN")
            st.info("Showing pre-saved forecast CSV.")
        except Exception as e: fc_err=str(e)
    else:
        # Load bundled forecast
        for p in ["outputs/results/future_forecast_30days.csv","future_forecast_30days.csv"]:
            if os.path.exists(p):
                try:
                    raw=pd.read_csv(p,parse_dates=["Date"])
                    fc_df=pd.DataFrame({"Date":pd.to_datetime(raw["Date"]),
                        "close":raw["Predicted_Close"],"pct":raw["Pct_vs_last"],
                        "pred_lr":0.,"day":range(1,len(raw)+1)})
                    fc_df["dir"]=fc_df["pct"].apply(lambda x:"UP" if float(x)>=0 else "DN")
                    break
                except Exception: pass

    if fc_err: st.error(f"Inference error: {fc_err}")

    left,right=st.columns([3,1],gap="medium")

    with left:
        if fc_df is not None and not fc_df.empty:
            lc=float(df_feat["Close"].iloc[-1]) if df_feat is not None else float(fc_df["close"].iloc[0])
            fp=float(fc_df["close"].iloc[-1]); fpct=float(fc_df["pct"].iloc[-1])
            dup=int((fc_df["dir"]=="UP").sum()); ddn=len(fc_df)-dup
            lds=df_feat.index[-1].strftime("%d %b %Y") if df_feat is not None else "—"

            c1,c2,c3,c4=st.columns(4)
            def mc(col,lbl,val,sub,cls=""):
                col.markdown(f"""<div class='metric-card'><div class='lbl'>{lbl}</div>
                  <div class='val {cls}'>{val}</div><div class='sub'>{sub}</div></div>""",
                  unsafe_allow_html=True)
            mc(c1,"LAST CLOSE",f"₹{lc:,.2f}",lds)
            arr="▲" if fpct>=0 else "▼"
            c2.markdown(f"""<div class='metric-card'><div class='lbl'>DAY {len(fc_df)} TARGET</div>
              <div class='val'>₹{fp:,.2f}</div>
              <div class='sub' style='color:{"#39FF14" if fpct>=0 else "#FF2D55"}'>{arr} {abs(fpct):.2f}% vs today</div>
              </div>""",unsafe_allow_html=True)
            mc(c3,"BULLISH DAYS",f"{dup}",f"of {len(fc_df)} days","lime")
            mc(c4,"BEARISH DAYS",f"{ddn}",f"of {len(fc_df)} days","red")
            st.markdown("<div style='height:.6rem'></div>",unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>PRICE TRAJECTORY</div>",unsafe_allow_html=True)
        rows=[]
        if df_raw is not None:
            t=df_raw["Close"].iloc[-hist_days:].reset_index(); t.columns=["Date","Price"]
            t["Series"]="Historical"; t["Dir"]="—"; rows.append(t)
        if fc_df is not None and not fc_df.empty:
            fp2=fc_df[["Date","close","dir"]].copy(); fp2.columns=["Date","Price","Dir"]; fp2["Series"]="Forecast"
            rows.append(fp2)
        if rows:
            pdf=pd.concat(rows,ignore_index=True); pdf["Date"]=pd.to_datetime(pdf["Date"])
            hl=alt.Chart(pdf[pdf["Series"]=="Historical"]).mark_line(color=CYAN,strokeWidth=1.3,opacity=.9).encode(
                x=alt.X("Date:T",axis=alt.Axis(format="%b %y",title="")),
                y=alt.Y("Price:Q",axis=alt.Axis(title="Price (₹)",format=",.0f")),
                tooltip=[alt.Tooltip("Date:T",format="%d %b %Y"),alt.Tooltip("Price:Q",format=",.2f",title="₹")])
            fl=alt.Chart(pdf[pdf["Series"]=="Forecast"]).mark_line(color=GOLD,strokeWidth=2,strokeDash=[5,2]).encode(
                x="Date:T",y="Price:Q")
            fd=alt.Chart(pdf[pdf["Series"]=="Forecast"]).mark_circle(size=55).encode(
                x="Date:T",y="Price:Q",
                color=alt.Color("Dir:N",scale=alt.Scale(domain=["UP","DN"],range=[LIME,RED]),
                    legend=alt.Legend(title="Direction")),
                tooltip=[alt.Tooltip("Date:T",format="%d %b %Y"),alt.Tooltip("Price:Q",format=",.2f",title="₹"),"Dir:N"])
            st.altair_chart(dark_alt(hl+fl+fd,h=310),use_container_width=True)

        if fc_df is not None and "pred_lr" in fc_df.columns and fc_df["pred_lr"].abs().sum()>0:
            st.markdown("<div class='sec-title'>LOG RETURN FORECAST</div>",unsafe_allow_html=True)
            lr=fc_df[["Date","pred_lr","dir"]].copy(); lr["lr_pct"]=lr["pred_lr"]*100
            bars=alt.Chart(lr).mark_bar(cornerRadiusTopLeft=2,cornerRadiusTopRight=2).encode(
                x=alt.X("Date:T",axis=alt.Axis(format="%d %b",title="")),
                y=alt.Y("lr_pct:Q",axis=alt.Axis(title="Log return (%)")),
                color=alt.Color("dir:N",scale=alt.Scale(domain=["UP","DN"],range=[LIME,RED]),legend=None),
                tooltip=[alt.Tooltip("Date:T",format="%d %b %Y"),
                         alt.Tooltip("lr_pct:Q",format=".4f",title="Log Ret %"),"dir:N"])
            z=alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color=MUTED,strokeDash=[4,3],opacity=.6).encode(y="y:Q")
            st.altair_chart(dark_alt(bars+z,h=160),use_container_width=True)

        if df_raw is not None:
            st.markdown("<div class='sec-title'>VOLUME & RETURN DISTRIBUTION</div>",unsafe_allow_html=True)
            hv1,hv2=st.columns(2)
            hist=df_raw.iloc[-hist_days:].copy().reset_index()
            hist.columns=["Date"]+list(hist.columns[1:]); hist["Date"]=pd.to_datetime(hist["Date"])
            hist["Dir"]=(hist["Close"]>=hist["Open"]).map({True:"UP",False:"DN"})
            hist["Ret"]=hist["Close"].pct_change()*100
            with hv1:
                vb=alt.Chart(hist).mark_bar(opacity=.78).encode(
                    x=alt.X("Date:T",axis=alt.Axis(format="%b %y",title="")),
                    y=alt.Y("Volume:Q",axis=alt.Axis(title="Volume")),
                    color=alt.Color("Dir:N",scale=alt.Scale(domain=["UP","DN"],range=[LIME,RED]),legend=None),
                    tooltip=[alt.Tooltip("Date:T",format="%d %b %Y"),alt.Tooltip("Volume:Q",format=",.0f")])
                st.altair_chart(dark_alt(vb,h=180),use_container_width=True)
            with hv2:
                rh=alt.Chart(hist.dropna(subset=["Ret"])).mark_bar(color=CYAN,opacity=.75).encode(
                    x=alt.X("Ret:Q",bin=alt.Bin(maxbins=50),axis=alt.Axis(title="Daily Return (%)")),
                    y=alt.Y("count()",axis=alt.Axis(title="Count")),
                    tooltip=[alt.Tooltip("Ret:Q",bin=True,title="Return %"),"count()"])
                mv=hist["Ret"].mean()
                mr=alt.Chart(pd.DataFrame({"x":[mv]})).mark_rule(color=GOLD,strokeDash=[4,3]).encode(x="x:Q")
                st.altair_chart(dark_alt(rh+mr,h=180),use_container_width=True)

    with right:
        st.markdown("<div class='sec-title'>DAY-BY-DAY</div>",unsafe_allow_html=True)
        if fc_df is not None and not fc_df.empty:
            st.markdown("<div class='fc-table'>",unsafe_allow_html=True)
            st.markdown("<div class='fc-row hdr'><span>DATE</span><span style='text-align:right;display:block'>₹</span><span style='text-align:right;display:block'>CHG%</span></div>",unsafe_allow_html=True)
            for _,row in fc_df.iterrows():
                d=pd.to_datetime(row["Date"]).strftime("%d %b")
                p=f"{float(row['close']):,.2f}"; pv=float(row["pct"])
                ps=f"+{pv:.2f}%" if pv>=0 else f"{pv:.2f}%"
                dc="up" if row["dir"]=="UP" else "dn"; ar="▲" if row["dir"]=="UP" else "▼"
                st.markdown(f"<div class='fc-row'><span class='d'>{d}</span><span class='p'>{p}</span><span class='c {dc}'>{ar} {ps}</span></div>",unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='upload-hint'>
              <div class='icon'>📡</div>
              <div class='msg'>Upload a stock CSV<br>to run live inference</div>
            </div>""",unsafe_allow_html=True)

        fp=f"{PLOT_DIR}/12_future_forecast_30d.png"
        if os.path.exists(fp):
            st.markdown("<div style='height:.6rem'></div>",unsafe_allow_html=True)
            plot_img(fp,"30-DAY FORECAST (TRAINING RUN)")

        st.markdown(f"<div class='disc'>⚠ NOT FINANCIAL ADVICE<br>Research prototype. Reliability degrades beyond day 5. Past performance ≠ future results.</div>",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════
elif TAB=="plots":
    st.markdown("<div class='sec-title'>TRAINING RUN — ANALYSIS PLOTS</div>",unsafe_allow_html=True)
    PLOTS=[
        (f"{PLOT_DIR}/01_eda_overview.png","EDA OVERVIEW — PRICE · VOLUME · DISTRIBUTIONS · CORRELATION"),
        (f"{PLOT_DIR}/04_fold_training_curves.png","WALK-FORWARD TRAINING CURVES — LOSS & MAE ACROSS 4 FOLDS"),
        (f"{PLOT_DIR}/06_test_actual_vs_predicted.png","ACTUAL vs PREDICTED — UNSEEN TEST SET (2021–2026)"),
        (f"{PLOT_DIR}/09_val_vs_test.png","VALIDATION vs TEST COMPARISON"),
        (f"{PLOT_DIR}/07_residuals.png","RESIDUALS — TEST SET"),
        (f"{PLOT_DIR}/08_scatter.png","ACTUAL vs PREDICTED SCATTER (₹)"),
        (f"{PLOT_DIR}/10_attention_weights.png","BAHDANAU ATTENTION WEIGHTS — TOP-5 ATTENDED DAYS (RED)"),
        (f"{PLOT_DIR}/11_metrics_summary.png","MODEL PERFORMANCE SUMMARY — MAPE & DIR. ACCURACY BY FOLD"),
        (f"{PLOT_DIR}/12_future_forecast_30d.png","30-DAY PRICE FORECAST"),
    ]
    full=[0,1,2,3,4,6,7,8]; side=[(5,None)]
    i=0
    while i<len(PLOTS):
        if i in full:
            plot_img(*PLOTS[i]); i+=1
        elif i==5:
            c1,c2=st.columns(2,gap="small")
            with c1: plot_img(*PLOTS[5])
            with c2:
                if 6<len(PLOTS): plot_img(*PLOTS[6])
            i=7
        else:
            plot_img(*PLOTS[i]); i+=1


# ══════════════════════════════════════════════════════════════════
#  TAB 3 — METRICS
# ══════════════════════════════════════════════════════════════════
elif TAB=="metrics":
    mdf=pd.DataFrame([
        {"Fold":"Fold 1 ★","MAPE_pct":4.942,"R2":0.9560,"DirAcc":54.43,"best":True},
        {"Fold":"Fold 2",  "MAPE_pct":6.040,"R2":0.8595,"DirAcc":51.05,"best":False},
        {"Fold":"Fold 3",  "MAPE_pct":3.742,"R2":0.8028,"DirAcc":50.11,"best":False},
        {"Fold":"Fold 4",  "MAPE_pct":4.850,"R2":0.9531,"DirAcc":52.11,"best":False},
    ])
    st.markdown("<div class='sec-title'>HOLD-OUT TEST SET</div>",unsafe_allow_html=True)
    def mc(col,lbl,val,sub,cls=""):
        col.markdown(f"""<div class='metric-card'><div class='lbl'>{lbl}</div>
          <div class='val {cls}'>{val}</div><div class='sub'>{sub}</div></div>""",unsafe_allow_html=True)
    m1,m2,m3,m4,m5=st.columns(5)
    mc(m1,"TEST MAPE","3.164%","Mean Absolute % Error","lime")
    mc(m2,"TEST R²","0.8799","Variance explained","cyan")
    mc(m3,"DIR ACC","50.45%","vs 50% random","")
    mc(m4,"EDGE","+0.45%","over random baseline","lime")
    mc(m5,"TEST RMSE","Rs.XX","Root Mean Sq Error","")

    st.markdown("<div style='height:.75rem'></div>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>FOLD COMPARISON</div>",unsafe_allow_html=True)
    fp=mdf.copy(); fp["Color"]=fp["best"].apply(lambda x:GOLD if x else CYAN)
    ch1,ch2,ch3=st.columns(3,gap="small")
    with ch1:
        b=alt.Chart(fp).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X("Fold:N",axis=alt.Axis(title="")),
            y=alt.Y("MAPE_pct:Q",axis=alt.Axis(title="MAPE %")),
            color=alt.Color("Color:N",scale=None,legend=None),
            tooltip=["Fold:N",alt.Tooltip("MAPE_pct:Q",format=".3f",title="MAPE %")]
        ).properties(title="MAPE %")
        st.altair_chart(dark_alt(b,h=240),use_container_width=True)
    with ch2:
        b2=alt.Chart(fp).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X("Fold:N",axis=alt.Axis(title="")),
            y=alt.Y("R2:Q",axis=alt.Axis(title="R²"),scale=alt.Scale(domain=[0,1])),
            color=alt.Color("Color:N",scale=None,legend=None),
            tooltip=["Fold:N",alt.Tooltip("R2:Q",format=".4f",title="R²")]
        ).properties(title="R² Score")
        st.altair_chart(dark_alt(b2,h=240),use_container_width=True)
    with ch3:
        r50=alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(color=RED,strokeDash=[5,3],opacity=.7).encode(y="y:Q")
        b3=alt.Chart(fp).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X("Fold:N",axis=alt.Axis(title="")),
            y=alt.Y("DirAcc:Q",axis=alt.Axis(title="Dir. Acc %"),scale=alt.Scale(domain=[45,60])),
            color=alt.Color("Color:N",scale=None,legend=None),
            tooltip=["Fold:N",alt.Tooltip("DirAcc:Q",format=".2f",title="Dir. Acc %")]
        ).properties(title="Directional Accuracy %")
        st.altair_chart(dark_alt(b3+r50,h=240),use_container_width=True)

    st.markdown("<div class='sec-title'>FULL TABLE</div>",unsafe_allow_html=True)
    d2=mdf[["Fold","MAPE_pct","R2","DirAcc"]].copy(); d2.columns=["Fold","MAPE %","R²","Dir. Acc %"]
    st.dataframe(d2.style.format({"MAPE %":"{:.3f}","R²":"{:.4f}","Dir. Acc %":"{:.2f}"}),
                 use_container_width=True,height=200)
    plot_img(f"{PLOT_DIR}/11_metrics_summary.png","MODEL PERFORMANCE SUMMARY (TRAINING RUN)")


# ══════════════════════════════════════════════════════════════════
#  TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
elif TAB=="arch":
    a1,a2=st.columns([1,1],gap="large")
    with a1:
        st.markdown("<div class='sec-title'>MODEL ARCHITECTURE</div>",unsafe_allow_html=True)
        LAYERS=[
            ("INPUT",    f"({cfg.get('sequence_len',60) if cfg else 60} × {cfg.get('n_features',13) if cfg else 13}) — OHLCV + 8 engineered features",CYAN),
            ("GRU 1",    f"{arch.get('gru1','?')} units · return_sequences=True · recurrent_dropout=0.05","#14B8A6"),
            ("LAYERNORM","Normalise activations across sequence",MUTED),
            ("GRU 2",    f"{arch.get('gru2','?')} units · return_sequences=True · recurrent_dropout=0.05","#14B8A6"),
            ("LAYERNORM","Normalise activations across sequence",MUTED),
            ("ATTENTION",f"Bahdanau additive · {arch.get('attn_units','?')} units → context vector",GOLD),
            ("DENSE 1",  f"{arch.get('dense1','?')} units · ReLU · Dropout({arch.get('dropout','?')})",CYAN),
            ("DENSE 2",  f"{arch.get('dense2','?')} units · ReLU · Dropout({float(arch.get('dropout',0.15))*0.5:.3f})",CYAN),
            ("OUTPUT",   "1 unit → Log-Return → inverse_transform → Price",LIME),
        ]
        for i,(name,desc,color) in enumerate(LAYERS):
            st.markdown(f"""<div class='arch-layer'>
              <div class='arch-box' style='color:{color};border-color:{color}44;background:{color}0D;'>{name}</div>
              <div class='arch-desc'>{desc}</div></div>
              {"<div class='arch-arrow'>↓</div>" if i<len(LAYERS)-1 else ""}""",unsafe_allow_html=True)

        st.markdown("<div class='sec-title' style='margin-top:1.25rem'>INPUT FEATURES</div>",unsafe_allow_html=True)
        feats=cfg.get("feature_cols",[]) if cfg else []
        st.markdown("".join([f"<span class='pill'>{f}</span>" for f in feats]),unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:.62rem;color:{MUTED};margin-top:.5rem;line-height:1.8;'>"
            f"TARGET: <span style='color:{GOLD}'>Log_Return</span> &nbsp;·&nbsp; "
            f"FEATURE SCALER: RobustScaler &nbsp;·&nbsp; TARGET SCALER: StandardScaler<br>"
            f"BEST FOLD: #{cfg.get('trained_on_fold','—') if cfg else '—'} — selected by minimum val_loss</div>",
            unsafe_allow_html=True)

    with a2:
        st.markdown("<div class='sec-title'>HYPERPARAMETERS</div>",unsafe_allow_html=True)
        HP=[("GRU UNITS",f"{arch.get('gru1','?')} / {arch.get('gru2','?')}"),
            ("ATTENTION UNITS",str(arch.get("attn_units","?"))),
            ("DENSE UNITS",f"{arch.get('dense1','?')} / {arch.get('dense2','?')}"),
            ("DROPOUT",str(arch.get("dropout","?"))),
            ("RECURRENT DROPOUT",str(arch.get("recurrent_dropout","?"))),
            ("L2 REGULARISATION",str(arch.get("l2","?"))),
            ("KERNEL INIT","glorot_uniform"),
            ("LOSS",str(tr.get("loss","?")).upper()),
            ("OPTIMISER","ADAM · clipnorm=1.0"),
            ("INIT LR",str(tr.get("init_lr","?"))),
            ("PEAK LR",str(tr.get("peak_lr","?"))),
            ("WARMUP EPOCHS",str(tr.get("warmup_epochs","?"))),
            ("MAX EPOCHS",str(tr.get("max_epochs","?"))),
            ("BATCH SIZE",str(tr.get("batch_size","?"))),
            ("EARLY STOP PAT.",str(tr.get("early_stopping_patience","?"))),
            ("SEQUENCE LENGTH",f"{cfg.get('sequence_len','?') if cfg else '?'} days"),
            ("TRAIN END DATE",str(cfg.get("train_end_date","?") if cfg else "?")),]
        st.markdown(f"<div style='background:{BG2};border:1px solid {LINE};border-radius:4px;overflow:hidden;'>",unsafe_allow_html=True)
        for k,v in HP:
            st.markdown(f"<div class='hp-row'><span class='hp-k'>{k}</span><span class='hp-v'>{v}</span></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

        st.markdown("<div class='sec-title' style='margin-top:1.25rem'>UNDERFITTING FIXES v2 → v4</div>",unsafe_allow_html=True)
        CHANGES=[("l2","1e-5","5e-6","too aggressive on small folds"),
            ("recurrent_dropout","0.1","0.05","blocking recurrent info flow"),
            ("dropout","0.2","0.15","reduced forward-pass noise"),
            ("gru1","128","160","more capacity for 30yr dynamics"),
            ("gru2","64","96","matched wider first layer"),
            ("kernel_init","he_normal","glorot_uniform","better for GRU gates"),
            ("batch_size","32","16","richer gradients on small folds"),
            ("max_epochs","100","150","Fold4 val falling at ep85"),
            ("patience","25","35","stop cutting off mid-convergence"),
            ("PEAK_LR","1e-3","8e-4","wider layers unstable at 1e-3"),
            ("best fold","last()","min(val_loss)","Fold4 was underfitting"),]
        st.markdown(f"<div style='background:{BG2};border:1px solid {LINE};border-radius:4px;overflow:hidden;'>",unsafe_allow_html=True)
        st.markdown(f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 2fr;padding:.35rem .75rem;font-size:.57rem;color:{MUTED};letter-spacing:.1em;background:{BG1};border-bottom:1px solid {LINE};'><span>PARAM</span><span>OLD</span><span>NEW</span><span>REASON</span></div>",unsafe_allow_html=True)
        for p,o,n,r in CHANGES:
            st.markdown(f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr 2fr;padding:.34rem .75rem;border-bottom:1px solid {GRID};font-size:.65rem;font-family:JetBrains Mono,monospace;'><span style='color:{DIM}'>{p}</span><span style='color:{RED}'>{o}</span><span style='color:{LIME}'>{n}</span><span style='color:{MUTED}'>{r}</span></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown(f"<div class='disc'>⚠ NOT FINANCIAL ADVICE · Research prototype only.<br>Past performance does not guarantee future results.</div>",unsafe_allow_html=True)

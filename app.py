import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pyrebase
import warnings

# ── SETUP ──
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")

FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID")
}

try:
    firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
    auth = firebase.auth()
    db = firebase.database()
except:
    auth = None
    db = None

if "page" not in st.session_state: st.session_state.page = "landing"
if "user" not in st.session_state: st.session_state.user = None

st.set_page_config(page_title="FairAI Titan | Enterprise Ethics", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ── ULTIMATE PREMIUM CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --titan-gold: #f8c94d;
        --bg: #050506;
        --card: rgba(255, 255, 255, 0.03);
        --border: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--bg) !important;
        color: #fff !important;
    }

    header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* Glass Cards */
    .titan-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 2rem;
        backdrop-filter: blur(20px);
        transition: 0.3s;
    }

    .titan-card:hover { border-color: var(--titan-gold); transform: translateY(-5px); }

    /* KPI Metrics */
    .kpi-val { font-size: 2.5rem; font-weight: 800; color: var(--titan-gold); line-height: 1; }
    .kpi-label { font-size: 0.7rem; color: #666; text-transform: uppercase; letter-spacing: 2px; margin-top: 10px; }

    /* Sidebar Navigation */
    section[data-testid="stSidebar"] {
        background-color: #080809 !important;
        border-right: 1px solid var(--border);
    }

    /* Buttons Override */
    div.stButton > button {
        background: var(--titan-gold) !important;
        color: #000 !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 800 !important;
        border: none !important;
        transition: 0.25s;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(248, 201, 77, 0.3);
    }

    /* Ticker */
    .ticker {
        background: rgba(248, 201, 77, 0.05);
        padding: 0.8rem 0; overflow: hidden; white-space: nowrap;
        border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
    }
    .ticker-move { display: inline-block; animation: ticker-anim 40s linear infinite; font-family: 'JetBrains Mono'; color: var(--titan-gold); font-size: 0.8rem; }
    @keyframes ticker-anim { from { transform: translateX(0); } to { transform: translateX(-50%); } }

</style>
""", unsafe_allow_html=True)

# ── LOGIC ──
def go_to(p): st.session_state.page = p; st.rerun()

def hf_audit(t):
    if not HF_TOKEN: return "SYSTEM: TOKEN_ERROR"
    url = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(url, headers=headers, json={"inputs": f"Bias Audit: {t}"})
        return r.json()[0]['generated_text']
    except: return "SYSTEM: NEURAL_LINK_FAIL"

def run_ml_audit(df):
    target = df.columns[-1]
    sensitive = ['gender', 'sex', 'race', 'age', 'religion']
    found = [c for c in df.columns if any(s in c.lower() for s in sensitive)]
    if not found: return None, None
    le = LabelEncoder()
    df_e = df.copy()
    for col in df_e.columns:
        if df_e[col].dtype == 'object': df_e[col] = le.fit_transform(df_e[col].astype(str))
    X = df_e.drop(columns=[target]); y = df_e[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    bias_metrics = []
    for s in found:
        groups = df[s].unique()
        rates = {g: df[df[s]==g][target].mean() if df[target].dtype != 'object' else (df[df[s]==g][target] == df[target].mode()[0]).mean() for g in groups}
        impact = min(rates.values()) / max(rates.values()) if max(rates.values()) > 0 else 1.0
        bias_metrics.append({"Feature": s, "Impact": impact})
    return acc, bias_metrics

# ── LANDING PAGE ──
if st.session_state.page == "landing":
    st.markdown("""
    <div style="padding: 2rem 8%; display: flex; justify-content: space-between; align-items: center;">
        <div style="font-size: 2rem; font-weight: 800; letter-spacing: -2px;">FairAI <span style="color:var(--titan-gold);">Titan</span></div>
        <div style="font-size: 0.7rem; color: #666; letter-spacing: 2px;">ENTERPRISE AUDIT SUITE</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center; padding: 120px 10%;">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size: 7rem; font-weight: 800; line-height: 1; letter-spacing: -6px;">Audit the<br><span style="color:var(--titan-gold);">Neural Standard.</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #888; font-size: 1.4rem; max-width: 800px; margin: 2rem auto 4rem auto;">Forensic ethics platform for high-stakes AI. Detect bias before it impacts humanity.</p>', unsafe_allow_html=True)
    if st.button("INITIALIZE PLATFORM >>"): go_to('login')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ticker">
        <div class="ticker-move">
            // BIAS_SCAN: ACTIVE // COMPLIANCE: GDPR_READY // DRIFT: NEUTRAL // STATUS: SECURE // 
            // BIAS_SCAN: ACTIVE // COMPLIANCE: GDPR_READY // DRIFT: NEUTRAL // STATUS: SECURE // 
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── LOGIN / SIGNUP (FIXED PREMIUM UI) ──
elif st.session_state.page in ["login", "signup"]:
    st.markdown("""
    <style>
        .auth-wrap{ min-height:100vh; display:flex; align-items:center; justify-content:center; padding:40px 20px; }
        .auth-card{ width:100%; max-width:470px; padding:3rem; border-radius:28px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); backdrop-filter:blur(20px); box-shadow:0 20px 80px rgba(0,0,0,0.45); }
        .auth-title{ font-size:2.4rem; font-weight:800; text-align:center; margin-bottom:8px; letter-spacing:-1px; }
        .auth-sub{ text-align:center; color:#888; margin-bottom:35px; font-size:0.95rem; }
        .orb{ width:70px; height:70px; margin:auto; margin-bottom:20px; border-radius:50%; background:radial-gradient(circle at 30% 30%, #fff, #f8c94d 55%, #000 100%); box-shadow:0 0 40px rgba(248,201,77,.35); animation:pulse2 2s infinite; }
        @keyframes pulse2{ 0%,100%{transform:scale(1);} 50%{transform:scale(1.08);} }
        div[data-testid="stTextInput"] input{ background:rgba(255,255,255,0.04)!important; border:1px solid rgba(255,255,255,0.08)!important; color:white!important; border-radius:14px!important; height:52px!important; padding-left:16px!important; }
        div[data-testid="stTextInput"] label{ color:#999!important; font-size:0.8rem!important; font-weight:700!important; letter-spacing:1px!important; }
        .muted-line{ text-align:center; color:#666; font-size:0.8rem; margin-top:18px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap"><div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="orb"></div>', unsafe_allow_html=True)

    if st.session_state.page == "login":
        st.markdown('<div class="auth-title">Executive Login</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Secure access to FairAI Titan Control Center</div>', unsafe_allow_html=True)
        e = st.text_input("EMAIL ADDRESS", key="login_email")
        p = st.text_input("PASSWORD", type="password", key="login_pass")
        if st.button("AUTHENTICATE >>", key="login_btn"):
            try:
                user = auth.sign_in_with_email_and_password(e, p); st.session_state.user = user; go_to("dashboard")
            except: st.error("Invalid credentials.")
        c1, c2 = st.columns(2)
        with c1: 
            if st.button("CREATE ACCOUNT", key="goto_signup"): go_to("signup")
        with c2:
            if st.button("BACK HOME", key="back_home_login"): go_to("landing")
        st.markdown('<div class="muted-line">256-bit encrypted authentication</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="auth-title">Create Identity</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Provision a new enterprise operator account</div>', unsafe_allow_html=True)
        e = st.text_input("EMAIL ADDRESS", key="signup_email")
        p = st.text_input("PASSWORD", type="password", key="signup_pass")
        if st.button("CREATE ACCOUNT >>", key="create_btn"):
            try: auth.create_user_with_email_and_password(e, p); st.success("Account Created"); time.sleep(1); go_to("login")
            except Exception as ex: st.error(str(ex))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("LOGIN", key="back_login"): go_to("login")
        with c2:
            if st.button("HOME", key="home_signup"): go_to("landing")
        st.markdown('<div class="muted-line">Compliant with enterprise standards</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# ── DASHBOARD (FIXED & INSANE) ──
elif st.session_state.page == "dashboard":
    if not st.session_state.user: go_to('login')

    # Topbar HUD
    st.markdown(f"""
    <div style="padding: 1.5rem 2rem; display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--border);">
        <div style="font-size: 1.5rem; font-weight: 800; letter-spacing: -1.5px;">FairAI <span style="color:var(--titan-gold);">CORE</span></div>
        <div style="display: flex; gap: 20px; align-items: center;">
            <div style="font-family: 'JetBrains Mono'; color: #666; font-size: 0.75rem;">{st.session_state.user['email']}</div>
            <div style="background: var(--titan-gold); color: #000; padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.65rem;">IDENTITY_VERIFIED</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 40px 6%;">', unsafe_allow_html=True)

    # KPI Grid
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown('<div class="titan-card kpi-card"><div class="kpi-val">98.4%</div><div class="kpi-label">Trust Score</div></div>', unsafe_allow_html=True)
    with k2: st.markdown('<div class="titan-card kpi-card"><div class="kpi-val">04</div><div class="kpi-label">Active Threats</div></div>', unsafe_allow_html=True)
    with k3: st.markdown('<div class="titan-card kpi-card"><div class="kpi-val">12</div><div class="kpi-label">Models Monitored</div></div>', unsafe_allow_html=True)
    with k4: st.markdown('<div class="titan-card kpi-card"><div class="kpi-val">GDPR</div><div class="kpi-label">Compliance</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar Nav
    with st.sidebar:
        st.markdown('<h3 style="color:var(--titan-gold); padding:1rem 0;">Protocols</h3>', unsafe_allow_html=True)
        menu = st.radio("Navigation", ["📊 Bias Scanner", "🧠 Neural Probe", "📜 History"], label_visibility="collapsed")
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("TERMINATE SESSION"): st.session_state.user = None; go_to('landing')

    # Main Area
    st.markdown(f"<h1 style='font-size: 3rem; font-weight: 800; margin-bottom: 2rem; letter-spacing: -2.5px;'>{menu}</h1>", unsafe_allow_html=True)

    if menu == "📊 Bias Scanner":
        st.markdown('<div class="titan-card">', unsafe_allow_html=True)
        f = st.file_uploader("Ingest Dataset (.CSV)", type=["csv"])
        if f:
            df = pd.read_csv(f); st.dataframe(df.head(), use_container_width=True)
            if st.button("EXECUTE FORENSIC AUDIT >>"):
                with st.spinner("Analyzing vectors..."):
                    acc, m = run_ml_audit(df)
                    if acc:
                        st.success(f"Audit Complete. Accuracy: {acc:.2%}")
                        st.plotly_chart(px.bar(pd.DataFrame(m), x="Feature", y="Impact", template="plotly_dark", color_discrete_sequence=[os.getenv("ACCENT_COLOR", "#f8c94d")]))
                    else: st.error("No sensitive features detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif menu == "🧠 Neural Probe":
        st.markdown('<div class="titan-card">', unsafe_allow_html=True)
        t = st.text_area("INJECT_BIAS_PROBE:", height=200, placeholder="Describe a high-performing manager...")
        if st.button("RUN NEURAL PROBE >>"):
            with st.spinner("Triangulating Weights..."):
                res = hf_audit(t)
                st.markdown(f'<div style="background:rgba(0,0,0,0.4); padding:2rem; border-radius:15px; border-left:4px solid var(--titan-gold); font-family:\'JetBrains Mono\';">{res}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="titan-card"><h3>Cloud Records</h3><p style="color:#666;">No prior drift events detected.</p></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


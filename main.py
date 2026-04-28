import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import time
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# ── SETUP ──
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairAI Inspector",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0d0d0f; color: #e8e6df; }
  .main-header {
    background: #111218;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
  }
</style>
""", unsafe_allow_html=True)

# ── DYNAMIC DATA GENERATORS (NO FIXED SEEDS) ──
def get_hiring_data(n=1000):
    # Use current time as seed for TRUE randomness
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32))
    gender = rng.choice(["Male", "Female"], n, p=[0.5, 0.5])
    exp = rng.integers(0, 15, n)
    score = rng.integers(40, 100, n)
    # Randomize the bias penalty intensity each time
    bias_penalty = rng.uniform(-30, -10) 
    merit = (score * 0.6 + exp * 3) + np.where(gender == "Female", bias_penalty, 0) + rng.normal(0, 10, n)
    hired = (merit > np.percentile(merit, 65)).astype(int)
    return pd.DataFrame({"gender": gender, "experience": exp, "score": score, "hired": hired})

def get_loan_data(n=1000):
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32))
    race = rng.choice(["White", "Minority"], n, p=[0.7, 0.3])
    income = rng.integers(30000, 120000, n)
    credit = rng.integers(500, 850, n)
    # Randomize racial bias intensity
    bias_penalty = rng.uniform(-60, -20)
    merit = (credit * 0.5 + income/1000 * 0.3) + np.where(race == "Minority", bias_penalty, 0) + rng.normal(0, 15, n)
    approved = (merit > np.percentile(merit, 75)).astype(int)
    return pd.DataFrame({"race": race, "income": income, "credit": credit, "approved": approved})

# ── CORE FUNCTIONS ──
def hf_audit(t, model_id):
    if not HF_TOKEN: return "Token Missing"
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(url, headers=headers, json={"inputs": f"Bias Audit: {t}"})
        return r.json()[0]['generated_text']
    except: return "Connection Error"

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚖️ FairAI Inspector")
    ds_choice = st.selectbox("Dataset", ["Hiring (Gender Bias)", "Loan Approval (Race Bias)"])
    ml_engine = st.selectbox("ML Engine", ["Logistic Regression", "Gradient Boosting"])
    st.markdown("---")
    st.markdown("### 🧠 Multi-Neural Hub")
    llm_model = st.selectbox("LLM Weight", ["google/gemma-2-2b-it", "meta-llama/Meta-Llama-3-8B-Instruct"])
    st.markdown("---")
    st.write("System Status: **LIVE_RE-SAMPLING**")
    run_btn = st.button("🔍 Run Live Audit", use_container_width=True)

# ── Header ──
st.markdown("""
<div class="main-header">
  <h1>⚖️ FairAI Inspector</h1>
  <p>Dynamic Neural Forensics & Live Bias Detection Engine.</p>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    st.info("👈 Click 'Run Live Audit' to trigger a fresh data re-sampling and ML training cycle.")
    st.stop()

# ── DYNAMIC PROCESSING ──
with st.spinner("Re-sampling Data & Re-training Model..."):
    # Clear Cache effectively by not using one
    if ds_choice == "Hiring (Gender Bias)":
        df = get_hiring_data()
        target, sensitive = "hired", "gender"
    else:
        df = get_loan_data()
        target, sensitive = "approved", "race"

    # Train Model
    le = LabelEncoder()
    df_e = df.copy()
    df_e[sensitive] = le.fit_transform(df[sensitive])
    
    X = df_e.drop(columns=[target])
    y = df_e[target]
    # Randomized split each time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    
    if ml_engine == "Logistic Regression":
        model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    else:
        model = GradientBoostingClassifier().fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Calculate Real Metrics from THIS run
    results = []
    groups = df[sensitive].unique()
    for g in groups:
        mask = df[sensitive] == g
        subset = df[mask]
        pos_rate = subset[target].mean()
        results.append({"Group": g, "Selection Rate": pos_rate})
    
    res_df = pd.DataFrame(results)
    gap = res_df["Selection Rate"].max() - res_df["Selection Rate"].min()

    # ── UI TABS ──
    tab1, tab2, tab3 = st.tabs(["🔴 Bias Report", "🧠 Neural Probe", "⚙️ Model Stats"])
    
    with tab1:
        st.write(f"### Live Parity Analysis ({sensitive})")
        st.write(f"Timestamp: `{time.ctime()}`")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#111218')
        ax.set_facecolor('#111218')
        # Use Dynamic Colors
        sns.barplot(data=res_df, x="Group", y="Selection Rate", palette="magma", ax=ax)
        ax.set_ylabel("Selection Rate (%)")
        ax.tick_params(colors='#e8e6df')
        st.pyplot(fig)
        
        st.metric("Detected Selection Gap", f"{gap*100:.1f}%", delta=f"{gap*100 - 20:.1f}% vs baseline", delta_color="inverse")
        
        if gap > 0.15:
            st.error(f"SYSTEM_ALERT: Significant bias detected in this sample.")
        else:
            st.warning("SYSTEM_ADVISORY: Minor parity drift observed.")

    with tab2:
        st.write(f"### Neural Probing Hub: {llm_model}")
        p = st.text_area("Audit Prompt:", placeholder="Enter a prompt to test...")
        if st.button("EXECUTE PROBE"):
            st.write(hf_audit(p, llm_model))

    with tab3:
        st.write("### Model Performance Metrics")
        st.metric("Test Accuracy", f"{acc:.2%}")
        st.write("### Dynamic Feature Correlations")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor('#111218')
        sns.heatmap(df_e.corr(), annot=True, cmap="YlGnBu", ax=ax2)
        st.pyplot(fig2)

st.markdown("---")
st.caption(f"FairAI Engine v4.0 | Entropy Source: {int(time.time())}")


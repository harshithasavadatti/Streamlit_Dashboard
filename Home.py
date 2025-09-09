import streamlit as st
import pandas as pd
from utils.load_data import load_data

# --- Page Configuration ---
st.set_page_config(
    page_title="🏦 Home Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --- Custom CSS for Dark Mode ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #141414 !important;
        color: #e0e0e0 !important;
    }
    .big-title {
        font-size:36px; font-weight:700; color:#F9AC40;
        letter-spacing:-2px; margin-bottom:0.3em;
    }
    .subtitle {
        font-size:22px; color:#52d1dc; font-weight:600;
        margin-bottom:0.1em;
    }
    .emoji-header {
        font-size:28px;
        margin:0 0 8px 0;
    }
    .info-card {
        background-color: #23272c;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom:16px;
        color: #e0e0e0;
        box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    }
    .nav-section {
        background: linear-gradient(93deg,#3a3f44 40%,#252829 100%);
        padding:10px 18px; border-radius:11px;
        color:#caf0f8; margin:10px 0;
    }
    .data-card {
        background:#15191c; border-radius:11px; padding:5px 12px;
        box-shadow: 0 1px 8px #fff2;
        margin-bottom: 18px
    }
    a { color: #F9AC40; text-decoration:underline;}
    hr {border:1px solid #2b2f32; margin:18px 0;}
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<div class="big-title">🏦 Home Credit Default Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ By Harshitha</div>', unsafe_allow_html=True)

# --- Welcome Cards (Highly Visual) ---
st.markdown("""
<div class="info-card">
🎉 <b>Welcome!</b> This dashboard lets you explore <span style='color:#F9AC40'>Home Credit’s loan risks</span>.<br>
Use the <b>sidebar</b> to discover drivers of default, visualize trends, and drill into 300K applications in seconds!
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nav-section">
🧭 <b>Navigate by:</b><br>
• 📊 <i>Overview & Data Quality</i><br>
• 🔎 <i>Target & Risk Segmentation</i> <br>
• 👨‍👩‍👧 <i>Demographics</i> <br>
• 💵 <i>Financial Health</i> <br>
• 🧬 <i>Correlations & Drivers</i>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Filters Preview ---
st.sidebar.markdown("## 🛠️ Filters")
st.sidebar.markdown("🎚️ Apply filters (gender and Education) to Interactive Analysis on Page 5")
st.sidebar.markdown("---")
st.sidebar.success("📢 Support dark mode!")

# --- Upload Section (Card) ---
st.markdown("""
<div class="info-card">
📤 <b>Upload Your Data</b>  
Drag & drop your <span style='color:#52d1dc'>File.csv</span> (or Excel format)—or explore using the built-in demo sample below.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Home Credit file (CSV/XLS/XLSX)", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"✅ Loaded `{uploaded_file.name}` ({df.shape[0]:,} rows, {df.shape[1]} columns)")
else:
    df = load_data()
    st.info("🗂️ Using demo dataset (10 rows previewed)", icon="📊")

st.markdown('<div class="data-card">', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Dashboard Features Highlight ---
st.markdown("""
<div class="info-card">
✨ <b>Features:</b><br>
🖥️ <b>5 Analysis Pages</b> · 💡 <b>10 KPIs and Charts per Page</b> · 🎛️ <b>Global Filters(Gender and Education)</b> · 📤 <b>Easy Data Export</b><br>
<b>Business insights are highlighted below every chart!</b>
</div>
""", unsafe_allow_html=True)

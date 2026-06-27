# =====================================================
# LIVIN NLP ANALYSIS DASHBOARD
# BAGIAN 1
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Livin NLP Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #FFFFFF;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #EAEAEA;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

h1,h2,h3 {
    color: #0F4C81;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("📊 Livin NLP Analysis Dashboard")

st.markdown("""
Dashboard ini digunakan untuk menampilkan proses NLP secara lengkap:

**Review → Cleaning → Case Folding → Normalization → Tokenization → Stopword Removal → Stemming → Teacher Sentiment → Teacher Emotion → IndoBERT Tokenizer → Embedding → Dashboard Analytics**
""")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    uploaded_file = st.file_uploader(
        "📁 Upload Dataset",
        type=["csv", "xlsx"]
    )

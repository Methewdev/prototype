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

# =====================================================
# LOAD DATASET
# =====================================================

df = None

if uploaded_file is not None:

    try:

        # Excel
        if uploaded_file.name.endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        # CSV
        else:

            try:

                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    on_bad_lines="skip"
                )

            except:

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    sep=",",
                    on_bad_lines="skip"
                )

        st.sidebar.success(
            f"✅ Dataset berhasil dimuat ({len(df)} review)"
        )

    except Exception as e:

        st.error(
            f"Gagal membaca file: {e}"
        )

        st.stop()

else:

    st.info(
        "📁 Silakan upload dataset terlebih dahulu."
    )

    st.stop()

# =====================================================
# AUTO DETECT REVIEW COLUMN
# =====================================================

review_candidates = [
    "content",
    "review",
    "comment",
    "ulasan",
    "text"
]

review_col = None

for col in review_candidates:

    if col in df.columns:

        review_col = col
        break

# fallback manual
if review_col is None:

    review_col = st.sidebar.selectbox(
        "Pilih Kolom Review",
        df.columns
    )

else:

    st.sidebar.success(
        f"📝 Kolom Review : {review_col}"
    )

# =====================================================
# DATASET INFO
# =====================================================

with st.sidebar:

    st.markdown("---")

    st.subheader("📌 Informasi Dataset")

    st.write(f"Baris : {len(df)}")
    st.write(f"Kolom : {len(df.columns)}")

# =====================================================
# TABS
# =====================================================

tab1 = st.tabs([
    "📊 Data Understanding"
])[0]

# =====================================================
# TAB 1
# =====================================================

# =====================================================
# TAB 1 : DATA UNDERSTANDING
# =====================================================

with tab1:

    st.subheader("📊 Data Understanding")

    # ==========================
    # Metrics
    # ==========================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Review",
        len(df)
    )

    col2.metric(
        "Jumlah Kolom",
        len(df.columns)
    )

    col3.metric(
        "Missing Value",
        int(df.isnull().sum().sum())
    )

    avg_length = (
        df[review_col]
        .astype(str)
        .apply(len)
        .mean()
    )

    col4.metric(
        "Rata-rata Panjang Review",
        round(avg_length)
    )

    st.markdown("---")

    # ==========================
    # Preview Dataset
    # ==========================

    st.subheader("📄 Preview Dataset")

    st.dataframe(
        df.head(10),
        width="stretch"
    )

    st.markdown("---")

    # ==========================
    # Informasi Kolom
    # ==========================

    st.subheader("📋 Informasi Kolom")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str)
    })

    st.dataframe(
        info_df,
        width="stretch"
    )

    st.markdown("---")

    # ==========================
    # Missing Value
    # ==========================

    st.subheader("📌 Missing Value")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isnull().sum()
    })

    st.dataframe(
        missing_df,
        width="stretch"
    )

    st.markdown("---")

    # ==========================
    # Distribusi Rating
    # ==========================

    if "score" in df.columns:

        st.subheader("⭐ Distribusi Rating")

        rating_count = (
            df["score"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_count.columns = [
            "Rating",
            "Total"
        ]

        fig = px.bar(
            rating_count,
            x="Rating",
            y="Total",
            text="Total",
            title="Distribusi Rating Pengguna"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )
# =====================================================
# SESSION STATE
# =====================================================

if "processed_df" not in st.session_state:

    st.session_state.processed_df = df

# =====================================================
# LIBRARY
# =====================================================

import re

from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)


# =====================================================
# Dictionary Normalisasi
# =====================================================

normalization_dict = {

    "gk": "tidak",
    "ga": "tidak",
    "yg": "yang",
    "bgt": "banget",
    "udh": "sudah",
    "tp": "tapi",
    "dr": "dari",
    "dgn": "dengan",
    "krn": "karena"

}

# =====================================================
# Stopword & Stemmer
# =====================================================

stop_factory = (
    StopWordRemoverFactory()
)

stop_words = set(
    stop_factory.get_stop_words()
)

stemmer = (
    StemmerFactory()
    .create_stemmer()
)


# =====================================================
# Cleaning Function
# =====================================================

def cleaning(text):

    text = str(text)

    text = re.sub(
        r"http\\S+",
        "",
        text
    )

    text = re.sub(
        r"www\\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\\s]",
        " ",
        text
    )

    text = re.sub(
        r"\\s+",
        " ",
        text
    )

    return text.strip()
	
# =====================================================
# Case Folding
# =====================================================

def case_folding(text):

    return text.lower()
	
	
# =====================================================
# Normalization
# =====================================================

def normalize(text):

    words = text.split()

    words = [

        normalization_dict.get(
            word,
            word
        )

        for word in words

    ]

    return " ".join(words)
	
	
# =====================================================
# Tokenization
# =====================================================

def tokenize(text):

    return text.split()
	
# =====================================================
# Stopword Removal
# =====================================================

def remove_stopwords(tokens):

    return [

        word

        for word in tokens

        if word not in stop_words

    ]

# =====================================================
# Stemming
# =====================================================

def stemming(tokens):

    return [

        stemmer.stem(word)

        for word in tokens

    ]
	
	
# =====================================================
# Buttonn Analis
# =====================================================


run_preprocessing = st.sidebar.button(
    "🚀 Jalankan Analisis"
)
with st.spinner("Teacher Sentiment..."):

    sentiment_result = process_df[
        "final_text"
    ].apply(
        predict_sentiment
    )

process_df = pd.concat(

    [
        process_df,
        sentiment_result
    ],

    axis=1
)
with st.spinner("Teacher Emotion..."):

    emotion_result = process_df[
        "final_text"
    ].apply(
        predict_emotion
    )

process_df = pd.concat(

    [
        process_df,
        emotion_result
    ],

    axis=1
)
st.session_state.processed_df = process_df
# =====================================================
# Generate Pipeline
# =====================================================

if run_preprocessing:

    process_df = df.copy()

    process_df["cleaning"] = (
        process_df[review_col]
        .apply(cleaning)
    )

    process_df["casefold"] = (
        process_df["cleaning"]
        .apply(case_folding)
    )

    process_df["normalisasi"] = (
        process_df["casefold"]
        .apply(normalize)
    )

    process_df["token"] = (
        process_df["normalisasi"]
        .apply(tokenize)
    )

    process_df["stopword"] = (
        process_df["token"]
        .apply(remove_stopwords)
    )

    process_df["stemming"] = (
        process_df["stopword"]
        .apply(stemming)
    )

    process_df["final_text"] = (
        process_df["stemming"]
        .apply(
            lambda x: " ".join(x)
        )
    )

    st.session_state.processed_df = (
        process_df
    )
# =====================================================
# Struktur Tabs
# =====================================================
tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([

    "🧹 Cleaning",

    "🔤 Case Folding",

    "📝 Normalization",

    "🔪 Tokenization",

    "🚫 Stopword Removal",

    "🌱 Stemming"
	
	"😊 Teacher Sentiment",

    "😡 Teacher Emotion"

])

# =====================================================
# Tab Cleaning
# =====================================================

with tab2:

    st.subheader(
        "Text Cleaning"
    )

    if "cleaning" in st.session_state.processed_df.columns:

        st.dataframe(

            st.session_state.processed_df[
                [
                    review_col,
                    "cleaning"
                ]
            ].head(20),

            use_container_width=True

        )

# =====================================================
# Case Folding
# =====================================================

with tab3:

    st.subheader(
        "Case Folding"
    )

    if "casefold" in st.session_state.processed_df.columns:

        st.dataframe(

            st.session_state.processed_df[
                [
                    "cleaning",
                    "casefold"
                ]
            ].head(20),

            use_container_width=True

        )
		
# =====================================================
# Normalization
# =====================================================
with tab4:

    st.subheader(
        "Normalization"
    )

    if "normalisasi" in st.session_state.processed_df.columns:

        st.dataframe(

            st.session_state.processed_df[
                [
                    "casefold",
                    "normalisasi"
                ]
            ].head(20),

            use_container_width=True

        )
# =====================================================
# Tokenization
# =====================================================
with tab5:

    st.subheader(
        "Tokenization"
    )

    if "token" in st.session_state.processed_df.columns:

        st.dataframe(

            st.session_state.processed_df[
                [
                    "normalisasi",
                    "token"
                ]
            ].head(20),

            use_container_width=True

        )
# =====================================================
# Stopword Removal
# =====================================================
with tab6:

    st.subheader(
        "Stopword Removal"
    )

    if "stopword" in st.session_state.processed_df.columns:

        st.dataframe(

            st.session_state.processed_df[
                [
                    "token",
                    "stopword"
                ]
            ].head(20),

            use_container_width=True

        )
# =====================================================
# Stemming
# =====================================================
with tab7:

    st.subheader(
        "Stemming"
    )

    if "final_text" in st.session_state.processed_df.columns:

        st.dataframe(

            st.session_state.processed_df[
                [
                    "stopword",
                    "stemming",
                    "final_text"
                ]
            ].head(20),

            use_container_width=True

        )
		
# =====================================================
# Teacher Sentiment
# =====================================================

with tab8:

    st.subheader("Teacher Sentiment")

    st.dataframe(

        st.session_state.processed_df[
            [
                "final_text",
                "sentiment",
                "sentiment_score"
            ]
        ],

        width="stretch"
    )
	
# =====================================================
# Teacher Emotion
# =====================================================
with tab9:

    st.subheader("Teacher Emotion")

    st.dataframe(

        st.session_state.processed_df[
            [
                "final_text",
                "emotion",
                "emotion_score"
            ]
        ],

        width="stretch"
    )

# =====================================================
# Library
# =====================================================
from transformers import pipeline

from deep_translator import GoogleTranslator

# =====================================================
# LOAD TEACHER SENTIMENT
# =====================================================

@st.cache_resource
def load_teacher_sentiment():

    return pipeline(
        "text-classification",
        model="mdhugol/indonesia-bert-sentiment-classification"
    )


# =====================================================
# LOAD TEACHER EMOTION
# =====================================================

@st.cache_resource
def load_teacher_emotion():

    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions"
    )
	
# =====================================================
# LOAD MODEL
# =====================================================

teacher_sentiment = load_teacher_sentiment()

teacher_emotion = load_teacher_emotion()


# =====================================================
# Sentiment Mapping
# =====================================================
sentiment_map = {

    "LABEL_0": "Positive",

    "LABEL_1": "Neutral",

    "LABEL_2": "Negative"

}

# =====================================================
# Sentiment Mapping
# =====================================================
emotion_map = {

    "joy": "Senang",

    "gratitude": "Senang",

    "approval": "Senang",

    "admiration": "Senang",

    "anger": "Marah",

    "annoyance": "Marah",

    "disapproval": "Marah",

    "sadness": "Sedih",

    "grief": "Sedih",

    "disappointment": "Sedih",

    "fear": "Frustasi",

    "frustration": "Frustasi",

    "confusion": "Frustasi"

}

# =====================================================
# Translate
# =====================================================

def translate(text):

    try:

        return GoogleTranslator(
            source="id",
            target="en"
        ).translate(text)

    except:

        return text
		
		
# =====================================================
# Predict Sentiment
# =====================================================


def predict_sentiment(text):

    result = teacher_sentiment(text)[0]

    return pd.Series({

        "sentiment":

        sentiment_map.get(
            result["label"],
            result["label"]
        ),

        "sentiment_score":

        round(
            result["score"],
            4
        )

    })
	
# =====================================================
# Predict Emotion
# =====================================================	

def predict_emotion(text):

    text = translate(text)

    result = teacher_emotion(text)[0]

    return pd.Series({

        "emotion":

        emotion_map.get(
            result["label"],
            result["label"]
        ),

        "emotion_score":

        round(
            result["score"],
            4
        )

    })
	
	

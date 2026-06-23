import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModel
)

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Livin NLP Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Livin NLP Pipeline Prototype")

st.markdown("""
Prototype untuk menunjukkan proses NLP:

Review
→ Cleaning
→ Case Folding
→ Normalization
→ Tokenization
→ Stopword Removal
→ Stemming
→ Tokenizer IndoBERT
→ Embedding IndoBERT
""")

# =====================================================
# LOAD INDOBERT
# =====================================================

@st.cache_resource
def load_indobert():

    tokenizer = AutoTokenizer.from_pretrained(
        "indobenchmark/indobert-base-p1"
    )

    model = AutoModel.from_pretrained(
        "indobenchmark/indobert-base-p1"
    )

    return tokenizer, model

# =====================================================
# NORMALIZATION DICTIONARY
# =====================================================

normalization_dict = {

    "gk": "tidak",
    "ga": "tidak",
    "yg": "yang",
    "bgt": "banget",
    "udh": "sudah",
    "tp": "tapi",
    "dgn": "dengan",
    "krn": "karena",
    "dr": "dari",
    "pd": "pada"

}

# =====================================================
# STOPWORD
# =====================================================

stop_factory = StopWordRemoverFactory()

stop_words = set(
    stop_factory.get_stop_words()
)

# =====================================================
# STEMMER
# =====================================================

stem_factory = StemmerFactory()

stemmer = stem_factory.create_stemmer()

# =====================================================
# PREPROCESSING FUNCTIONS
# =====================================================

def cleaning(text):

    text = str(text)

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def case_folding(text):

    return text.lower()


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


def tokenize(text):

    return text.split()


def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word not in stop_words
    ]


def stemming(tokens):

    return [
        stemmer.stem(word)
        for word in tokens
    ]

# =====================================================
# UPLOAD FILE
# =====================================================

uploaded_file = st.file_uploader(
    "Upload CSV / XLSX",
    type=["csv", "xlsx"]
)

if uploaded_file:

    try:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        st.success(
            "Dataset berhasil diupload"
        )

    except Exception as e:

        st.error(
            f"Error : {e}"
        )

        st.stop()

    review_col = st.selectbox(
        "Pilih Kolom Review",
        df.columns
    )

    # =================================================
    # PREPROCESSING
    # =================================================

    df["cleaning"] = df[
        review_col
    ].apply(cleaning)

    df["casefold"] = df[
        "cleaning"
    ].apply(case_folding)

    df["normalisasi"] = df[
        "casefold"
    ].apply(normalize)

    df["token"] = df[
        "normalisasi"
    ].apply(tokenize)

    df["stopword"] = df[
        "token"
    ].apply(remove_stopwords)

    df["stemming"] = df[
        "stopword"
    ].apply(stemming)

    df["final_text"] = df[
        "stemming"
    ].apply(
        lambda x: " ".join(x)
    )

    # =================================================
    # TABS
    # =================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([

        "📊 Data Understanding",
        "🧹 Preprocessing",
        "🤖 Tokenizer IndoBERT",
        "🔢 Embedding IndoBERT",
        "📈 Dashboard"

    ])

    # =================================================
    # TAB 1
    # =================================================

    with tab1:

        st.subheader(
            "Dataset Overview"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Jumlah Review",
            len(df)
        )

        col2.metric(
            "Jumlah Kolom",
            len(df.columns)
        )

        st.dataframe(
            df.head()
        )

    # =================================================
    # TAB 2
    # =================================================

    with tab2:

        st.subheader(
            "Hasil Preprocessing"
        )

        st.dataframe(

            df[
                [
                    review_col,
                    "cleaning",
                    "casefold",
                    "normalisasi",
                    "token",
                    "stopword",
                    "stemming",
                    "final_text"
                ]
            ].head(20)

        )

    # =================================================
    # TAB 3
    # =================================================

    with tab3:

        tokenizer, model = (
            load_indobert()
        )

        sample = df[
            "final_text"
        ].iloc[0]

        st.subheader(
            "Sample Review"
        )

        st.code(sample)

        st.subheader(
            "Tokens"
        )

        st.write(
            tokenizer.tokenize(
                sample
            )
        )

        encoded = tokenizer(
            sample,
            truncation=True,
            max_length=128
        )

        st.subheader(
            "Input IDs"
        )

        st.write(
            encoded["input_ids"]
        )

        st.subheader(
            "Attention Mask"
        )

        st.write(
            encoded["attention_mask"]
        )

    # =================================================
    # TAB 4
    # =================================================

    with tab4:

        inputs = tokenizer(
            sample,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        embedding = (
            outputs.last_hidden_state
        )

        st.subheader(
            "Embedding Shape"
        )

        st.write(
            embedding.shape
        )

        st.info("""
        Penjelasan:

        Baris pertama:
        jumlah token

        Kolom:
        768 dimensi embedding IndoBERT
        """)

        emb_df = pd.DataFrame(

            embedding[0]
            .cpu()
            .numpy()

        )

        st.subheader(
            "Embedding Matrix"
        )

        st.dataframe(
            emb_df
        )

    # =================================================
    # TAB 5
    # =================================================

    with tab5:

        st.subheader(
            "Panjang Review"
        )

        df["length"] = df[
            review_col
        ].astype(str).apply(len)

        fig = px.histogram(
            df,
            x="length",
            nbins=20,
            title="Distribusi Panjang Review"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Top 20 Kata"
        )

        words = " ".join(
            df["final_text"]
        ).split()

        word_freq = pd.Series(
            words
        ).value_counts()

        st.dataframe(
            word_freq
            .head(20)
            .reset_index()
            .rename(
                columns={
                    "index":"Kata",
                    0:"Frekuensi"
                }
            )
        )

        st.success("""
        Pipeline NLP berhasil dijalankan
        hingga Embedding IndoBERT.
        """)

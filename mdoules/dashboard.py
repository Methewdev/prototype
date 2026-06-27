"""
=====================================================
DASHBOARD MODULE
=====================================================
Dashboard Analytics
=====================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# METRIC CARD
# =====================================================

def dashboard_metrics(df):

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Review",
        len(df)
    )

    col2.metric(
        "Sentiment Dominan",
        df["teacher_sentiment"].mode()[0]
    )

    col3.metric(
        "Emosi Dominan",
        df["teacher_emotion"].mode()[0]
    )

    col4.metric(
        "Rata-rata Panjang Review",
        round(
            df["final_text"]
            .str.len()
            .mean()
        )
    )


# =====================================================
# PIE SENTIMENT
# =====================================================

def sentiment_chart(df):

    sentiment = (

        df["teacher_sentiment"]

        .value_counts()

        .reset_index()

    )

    sentiment.columns = [

        "Sentiment",

        "Jumlah"

    ]

    fig = px.pie(

        sentiment,

        names="Sentiment",

        values="Jumlah",

        hole=0.4,

        title="Distribusi Sentiment"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# BAR EMOTION
# =====================================================

def emotion_chart(df):

    emotion_order = [
        "Senang",
        "Netral",
        "Marah",
        "Frustasi"
    ]

    emotion = (

        df["teacher_emotion"]

        .value_counts()

        .reindex(
            emotion_order,
            fill_value=0
        )

        .reset_index()

    )

    emotion.columns = [

        "Emotion",

        "Jumlah"

    ]

    fig = px.bar(

        emotion,

        x="Emotion",

        y="Jumlah",

        text="Jumlah",

        title="Distribusi Emosi"

    )

    fig.update_layout(

        xaxis_title="Emosi",

        yaxis_title="Jumlah Review"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# CROSSTAB
# =====================================================

def sentiment_vs_emotion(df):

    emotion_order = [

        "Senang",

        "Netral",

        "Marah",

        "Frustasi"

    ]

    cross = pd.crosstab(

        df["teacher_sentiment"],

        df["teacher_emotion"]

    )

    cross = cross.reindex(

        columns=emotion_order,

        fill_value=0

    )

    st.dataframe(
        cross,
        width="stretch"
    )


# =====================================================
# TOP WORDS
# =====================================================

def top_words(df):

    words = " ".join(
        df["final_text"]
    ).split()

    freq = (

        pd.Series(words)

        .value_counts()

        .head(20)

        .reset_index()

    )

    freq.columns = [

        "Kata",

        "Frekuensi"

    ]

    fig = px.bar(

        freq,

        x="Kata",

        y="Frekuensi",

        text="Frekuensi",

        title="20 Kata Terbanyak"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# DOWNLOAD
# =====================================================

def download_result(df):

    st.download_button(

        label="📥 Download Hasil Analisis",

        data=df.to_csv(index=False),

        file_name="hasil_analisis.csv",

        mime="text/csv"

    )

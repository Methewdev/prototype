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
# DASHBOARD METRICS
# =====================================================

def dashboard_metrics(df):

    if df is None:
        st.warning("Belum ada data hasil analisis.")
        return

    if df.empty:
        st.warning("Dataset kosong.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Review",
        len(df)
    )

    col2.metric(
        "Sentiment Dominan",
        df["teacher_sentiment"].mode().iloc[0]
    )

    col3.metric(
        "Emosi Dominan",
        df["teacher_emotion"].mode().iloc[0]
    )

    col4.metric(
        "Rata-rata Panjang Review",
        round(
            df["final_text"].astype(str).str.len().mean()
        )
    )


# =====================================================
# PIE CHART SENTIMENT
# =====================================================

def sentiment_chart(df):

    st.subheader("🥧 Distribusi Sentiment")

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
        hole=0.45,
        title="Distribusi Sentiment"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# BAR CHART EMOTION
# =====================================================

def emotion_chart(df):

    st.subheader("😊 Distribusi Emosi")

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
        "Emosi",
        "Jumlah"
    ]

    fig = px.bar(
        emotion,
        x="Emosi",
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

    st.subheader("📊 Crosstab Sentiment vs Emotion")

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
# STATISTIK SENTIMENT
# =====================================================

def sentiment_table(df):

    st.subheader("📈 Statistik Sentiment")

    sentiment = (
        df["teacher_sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment.columns = [
        "Sentiment",
        "Jumlah"
    ]

    sentiment["Persentase (%)"] = (
        sentiment["Jumlah"]
        / sentiment["Jumlah"].sum()
        * 100
    ).round(2)

    sentiment.insert(
        0,
        "No",
        range(1, len(sentiment) + 1)
    )

    st.dataframe(
        sentiment,
        width="stretch",
        hide_index=True
    )


# =====================================================
# STATISTIK EMOTION
# =====================================================

def emotion_table(df):

    st.subheader("📈 Statistik Emosi")

    emotion = (
        df["teacher_emotion"]
        .value_counts()
        .reset_index()
    )

    emotion.columns = [
        "Emosi",
        "Jumlah"
    ]

    emotion["Persentase (%)"] = (
        emotion["Jumlah"]
        / emotion["Jumlah"].sum()
        * 100
    ).round(2)

    emotion.insert(
        0,
        "No",
        range(1, len(emotion) + 1)
    )

    st.dataframe(
        emotion,
        width="stretch",
        hide_index=True
    )


# =====================================================
# TOP 20 WORDS
# =====================================================

def top_words(df):

    st.subheader("🔥 20 Kata Terbanyak")

    words = " ".join(
        df["final_text"].astype(str)
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
        title="Top 20 Words"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# PREVIEW HASIL ANALISIS
# =====================================================

def preview_result(df):

    st.subheader("📄 Preview Hasil Analisis")

    # Copy dataframe
    preview = df.copy()

    # Tambahkan nomor urut
    preview.insert(
        0,
        "No",
        range(1, len(preview) + 1)
    )

    # Ubah nama kolom agar lebih mudah dibaca
    preview = preview.rename(
        columns={

            "userName": "User",

            "score": "Rating",

            "content": "Review",

            "tanggal": "Tanggal",

            "cleaning": "Cleaning",

            "casefold": "Case Folding",

            "normalisasi": "Normalisasi",

            "token": "Token",

            "stopword": "Stopword",

            "stemming": "Stemming",

            "final_text": "Final Text",

            "teacher_sentiment": "Sentiment",

            "sentiment_score": "Sentiment Score",

            "teacher_emotion": "Emotion",

            "emotion_score": "Emotion Score"

        }
    )

    st.write(f"**Total Data : {len(preview):,} Review**")

    st.dataframe(

        preview,

        width="stretch",

        hide_index=True,

        column_config={

            "No": st.column_config.NumberColumn(
                width="small"
            ),

            "User": st.column_config.TextColumn(
                width="medium"
            ),

            "Rating": st.column_config.NumberColumn(
                width="small"
            ),

            "Review": st.column_config.TextColumn(
                width="large"
            ),

            "Tanggal": st.column_config.TextColumn(
                width="medium"
            ),

            "Cleaning": st.column_config.TextColumn(
                width="large"
            ),

            "Case Folding": st.column_config.TextColumn(
                width="large"
            ),

            "Normalisasi": st.column_config.TextColumn(
                width="large"
            ),

            "Token": st.column_config.TextColumn(
                width="large"
            ),

            "Stopword": st.column_config.TextColumn(
                width="large"
            ),

            "Stemming": st.column_config.TextColumn(
                width="large"
            ),

            "Final Text": st.column_config.TextColumn(
                width="large"
            ),

            "Sentiment": st.column_config.TextColumn(
                width="small"
            ),

            "Sentiment Score": st.column_config.NumberColumn(
                format="%.4f"
            ),

            "Emotion": st.column_config.TextColumn(
                width="small"
            ),

            "Emotion Score": st.column_config.NumberColumn(
                format="%.4f"
            )

        }

    )
# =====================================================
# DOWNLOAD CSV
# =====================================================

def download_result(df):

    download_df = df.copy()

    download_df.insert(
        0,
        "No",
        range(1, len(download_df)+1)
    )

    download_df = download_df.rename(
        columns={
            "userName":"User",
            "score":"Rating",
            "content":"Review",
            "tanggal":"Tanggal",
            "cleaning":"Cleaning",
            "casefold":"Case Folding",
            "normalisasi":"Normalisasi",
            "token":"Token",
            "stopword":"Stopword",
            "stemming":"Stemming",
            "final_text":"Final Text",
            "teacher_sentiment":"Sentiment",
            "sentiment_score":"Sentiment Score",
            "teacher_emotion":"Emotion",
            "emotion_score":"Emotion Score"
        }
    )

    csv = download_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "📥 Download Hasil Analisis",

        csv,

        "hasil_analisis.csv",

        "text/csv"

    )

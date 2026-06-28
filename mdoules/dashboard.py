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
st.write(type(process_df))
st.write(process_df)
    # ==============================
    # DEBUG
    # ==============================

    st.write("### DEBUG Dashboard")

    st.write("Type :", type(df))

    if df is None:

        st.error("processed_df = None")

        return

    if not isinstance(df, pd.DataFrame):

        st.error(
            f"processed_df bukan DataFrame tetapi {type(df)}"
        )

        st.write(df)

        return

    if df.empty:

        st.warning("DataFrame kosong")

        return

    # ==============================
    # Metric
    # ==============================

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
            df["final_text"]
            .str.len()
            .mean()
        )
    )

# =====================================================
# PIE CHART SENTIMENT
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
# CROSSTAB SENTIMENT vs EMOTION
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

    st.subheader("📊 Crosstab Sentiment vs Emotion")

    st.dataframe(
        cross,
        width="stretch"
    )


# =====================================================
# STATISTIK SENTIMENT
# =====================================================

def sentiment_table(df):

    st.subheader("📋 Statistik Sentiment")

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
        sentiment["Jumlah"] /
        sentiment["Jumlah"].sum()
        * 100
    ).round(2)

    sentiment.insert(
        0,
        "No",
        range(1, len(sentiment)+1)
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

    st.subheader("😊 Statistik Emosi")

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
        emotion["Jumlah"] /
        emotion["Jumlah"].sum()
        * 100
    ).round(2)

    emotion.insert(
        0,
        "No",
        range(1, len(emotion)+1)
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
# PREVIEW HASIL ANALISIS
# =====================================================

def preview_result(df):

    st.subheader("📄 Hasil Analisis")

    preview = df[
        [
            "userName",
            "score",
            "content",
            "teacher_sentiment",
            "teacher_emotion"
        ]
    ].copy()

    preview.columns = [
        "Username",
        "Rating",
        "Review",
        "Sentiment",
        "Emotion"
    ]

    preview.insert(
        0,
        "No",
        range(1, len(preview)+1)
    )

    st.write(
        f"Total Data : **{len(preview):,} Review**"
    )

    st.dataframe(
        preview,
        width="stretch",
        hide_index=True
    )


# =====================================================
# DOWNLOAD CSV
# =====================================================

def download_result(df):

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil Analisis",
        data=csv,
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )

# =====================================================
# PIE CHART SENTIMENT
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
# CROSSTAB SENTIMENT vs EMOTION
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

    st.subheader("📊 Crosstab Sentiment vs Emotion")

    st.dataframe(
        cross,
        width="stretch"
    )


# =====================================================
# STATISTIK SENTIMENT
# =====================================================

def sentiment_table(df):

    st.subheader("📋 Statistik Sentiment")

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
        sentiment["Jumlah"] /
        sentiment["Jumlah"].sum()
        * 100
    ).round(2)

    sentiment.insert(
        0,
        "No",
        range(1, len(sentiment)+1)
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

    st.subheader("😊 Statistik Emosi")

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
        emotion["Jumlah"] /
        emotion["Jumlah"].sum()
        * 100
    ).round(2)

    emotion.insert(
        0,
        "No",
        range(1, len(emotion)+1)
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
# PREVIEW HASIL ANALISIS
# =====================================================

def preview_result(df):

    st.subheader("📄 Hasil Analisis")

    preview = df[
        [
            "userName",
            "score",
            "content",
            "teacher_sentiment",
            "teacher_emotion"
        ]
    ].copy()

    preview.columns = [
        "Username",
        "Rating",
        "Review",
        "Sentiment",
        "Emotion"
    ]

    preview.insert(
        0,
        "No",
        range(1, len(preview)+1)
    )

    st.write(
        f"Total Data : **{len(preview):,} Review**"
    )

    st.dataframe(
        preview,
        width="stretch",
        hide_index=True
    )


# =====================================================
# DOWNLOAD CSV
# =====================================================

def download_result(df):

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil Analisis",
        data=csv,
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )

"""
=====================================================
DASHBOARD
=====================================================
"""

import streamlit as st
import plotly.express as px
import pandas as pd

# =====================================================
# SENTIMENT
# =====================================================

def sentiment_chart(df):

    fig = px.pie(

        df,

        names="teacher_sentiment",

        title="Distribusi Sentiment"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# =====================================================
# EMOTION
# =====================================================

def emotion_chart(df):

    emo = (

        df["teacher_emotion"]

        .value_counts()

        .reset_index()

    )

    emo.columns = [

        "Emotion",

        "Jumlah"

    ]

    fig = px.bar(

        emo,

        x="Emotion",

        y="Jumlah",

        text="Jumlah"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# =====================================================
# CROSSTAB
# =====================================================

def sentiment_vs_emotion(df):

    cross = pd.crosstab(

        df["teacher_sentiment"],

        df["teacher_emotion"]

    )

    st.dataframe(
        cross,
        width="stretch"
    )

# =====================================================
# TOP WORD
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

        "Word",

        "Frequency"

    ]

    fig = px.bar(

        freq,

        x="Word",

        y="Frequency",

        text="Frequency"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

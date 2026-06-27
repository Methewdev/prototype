"""
=====================================================
TEACHER MODEL
=====================================================
1. Teacher Sentiment
2. Teacher Emotion
=====================================================
"""

import pandas as pd
import streamlit as st

from transformers import pipeline

from deep_translator import GoogleTranslator

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_teacher_models():

    sentiment = pipeline(
        "text-classification",
        model="mdhugol/indonesia-bert-sentiment-classification"
    )

    emotion = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions"
    )

    return sentiment, emotion

# =====================================================
# MAPPING SENTIMENT
# =====================================================

SENTIMENT_MAP = {

    "LABEL_0": "Positive",

    "LABEL_1": "Neutral",

    "LABEL_2": "Negative"

}

# =====================================================
# MAPPING EMOTION
# =====================================================

EMOTION_MAP = {

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
# TRANSLATE
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
# PIPELINE
# =====================================================

def teacher_pipeline(df):

    sentiment_model, emotion_model = load_teacher_models()

    sentiments = []
    sentiment_scores = []

    emotions = []
    emotion_scores = []

    progress = st.progress(0)

    total = len(df)

    for i, text in enumerate(df["final_text"]):

        # ==========================
        # Sentiment
        # ==========================

        s = sentiment_model(text)[0]

        sentiments.append(

            SENTIMENT_MAP.get(
                s["label"],
                s["label"]
            )

        )

        sentiment_scores.append(

            round(
                s["score"],
                4
            )

        )

        # ==========================
        # Emotion
        # ==========================

        en_text = translate(text)

        e = emotion_model(en_text)[0]

        emotions.append(

            EMOTION_MAP.get(
                e["label"],
                e["label"]
            )

        )

        emotion_scores.append(

            round(
                e["score"],
                4
            )

        )

        progress.progress(
            (i + 1) / total
        )

    df["teacher_sentiment"] = sentiments

    df["sentiment_score"] = sentiment_scores

    df["teacher_emotion"] = emotions

    df["emotion_score"] = emotion_scores

    progress.empty()

    return df

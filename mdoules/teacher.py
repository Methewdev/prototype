"""
=====================================================
TEACHER MODEL
=====================================================
1. Teacher Sentiment
2. Teacher Emotion
=====================================================
"""

import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_teacher_models():

    sentiment_model = pipeline(
        "text-classification",
        model="mdhugol/indonesia-bert-sentiment-classification"
    )

    emotion_model = pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions"
    )

    return sentiment_model, emotion_model


# =====================================================
# SENTIMENT MAP
# =====================================================

SENTIMENT_MAP = {
    "LABEL_0": "Positive",
    "LABEL_1": "Neutral",
    "LABEL_2": "Negative"
}


# =====================================================
# EMOTION MAP
# =====================================================

EMOTION_MAP = {

    # Senang
    "joy": "Senang",
    "admiration": "Senang",
    "approval": "Senang",
    "gratitude": "Senang",
    "love": "Senang",
    "optimism": "Senang",
    "relief": "Senang",
    "pride": "Senang",
    "amusement": "Senang",
    "caring": "Senang",
    "desire": "Senang",
    "excitement": "Senang",

    # Marah
    "anger": "Marah",
    "annoyance": "Marah",
    "disapproval": "Marah",
    "disgust": "Marah",

    # Frustasi
    "fear": "Frustasi",
    "confusion": "Frustasi",
    "disappointment": "Frustasi",
    "sadness": "Frustasi",
    "grief": "Frustasi",
    "embarrassment": "Frustasi",
    "nervousness": "Frustasi",
    "remorse": "Frustasi",

    # Netral
    "neutral": "Netral",
    "realization": "Netral",
    "curiosity": "Netral",
    "surprise": "Netral"
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

    except Exception:

        return text


# =====================================================
# PREDICT SENTIMENT
# =====================================================

def predict_sentiment(text, sentiment_model):

    result = sentiment_model(text)[0]

    sentiment = SENTIMENT_MAP.get(
        result["label"],
        "Neutral"
    )

    score = round(result["score"], 4)

    return sentiment, score


# =====================================================
# PREDICT EMOTION
# =====================================================

def predict_emotion(text, emotion_model):

    english = translate(text)

    result = emotion_model(english)[0]

    emotion = EMOTION_MAP.get(
        result["label"],
        "Netral"
    )

    score = round(result["score"], 4)

    return emotion, score


# =====================================================
# TEACHER PIPELINE
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

        # -----------------------------
        # Sentiment
        # -----------------------------
        sentiment, sentiment_score = predict_sentiment(
            text,
            sentiment_model
        )

        sentiments.append(sentiment)
        sentiment_scores.append(sentiment_score)

        # -----------------------------
        # Emotion
        # -----------------------------
        emotion, emotion_score = predict_emotion(
            text,
            emotion_model
        )

        emotions.append(emotion)
        emotion_scores.append(emotion_score)

        progress.progress((i + 1) / total)

    progress.empty()

    df["teacher_sentiment"] = sentiments
    df["sentiment_score"] = sentiment_scores

    df["teacher_emotion"] = emotions
    df["emotion_score"] = emotion_scores

    return df

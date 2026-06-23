# =====================================================
# INUT LIBRARY
# ===================================================== 
import streamlit as st
import pandas as pd
import numpy as np
import re
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel
)

from deep_translator import GoogleTranslator

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.decomposition import PCA

import plotly.express as px

# =====================================================
# Load Teacher Sentiment
# ===================================================== 
@st.cache_resource
def load_sentiment_teacher():

    return pipeline(
        "text-classification",
        model="mdhugol/indonesia-bert-sentiment-classification"
    )
	
# =====================================================
# Load Teacher Emotion
# ===================================================== 
@st.cache_resource
def load_emotion_teacher():

    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions"
    )
	
# =====================================================
# Load IndoBERT
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
# Mapping Sentiment
# ===================================================== 

sentiment_map = {
    "LABEL_0": "Positive",
    "LABEL_1": "Neutral",
    "LABEL_2": "Negative"
}

# =====================================================
# Mapping Emotion
# ===================================================== 

emotion_mapping = {

    "joy": "Senang",
    "admiration": "Senang",
    "approval": "Senang",
    "gratitude": "Senang",

    "anger": "Marah",
    "annoyance": "Marah",
    "disapproval": "Marah",

    "sadness": "Sedih",
    "grief": "Sedih",
    "disappointment": "Sedih",

    "frustration": "Frustasi",
    "confusion": "Frustasi",
    "fear": "Frustasi"

}

# =====================================================
# Preprocessing
# ===================================================== 
def cleaning(text):

    text = str(text)

    text = re.sub(r"http\S+","",text)

    text = re.sub(r"www\S+","",text)

    text = re.sub(r"[^a-zA-Z\s]"," ",text)

    text = re.sub(r"\s+"," ",text)

    return text.strip()
	
# =====================================================
# Case Folding
# ===================================================== 
def case_folding(text):
    return text.lower()
	
# =====================================================
# Normalization
# ===================================================== 	

normalization_dict = {
    "gk":"tidak",
    "ga":"tidak",
    "yg":"yang",
    "bgt":"banget",
    "udh":"sudah"
}

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
# Tokenize
# ===================================================== 
def tokenize(text):
    return text.split()
	
# =====================================================
# Stop_Word
# ===================================================== 

stop_words = set(
    StopWordRemoverFactory()
    .get_stop_words()
)

# =====================================================
# Remove_Word
# ===================================================== 
def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word not in stop_words
    ]

# =====================================================
# Stemmer
# ===================================================== 

stemmer = (
    StemmerFactory()
    .create_stemmer()
)

def stemming(tokens):

    return [
        stemmer.stem(word)
        for word in tokens
    ]
	
	
# =====================================================
# Upload Dataset
# ===================================================== 
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    review_col = st.selectbox(
        "Pilih Kolom Review",
        df.columns
    )

    st.dataframe(df.head())

else:

    st.info(
        "Silakan upload dataset terlebih dahulu"
    )

    st.stop()
# =====================================================
# Pilih Kolom Review
# ===================================================== 

review_col = st.selectbox(
    "Pilih Kolom Review",
    df.columns
)

# =====================================================
# Generate Report
# ===================================================== 

df["cleaning"] = df[review_col].apply(cleaning)

df["casefold"] = df["cleaning"].apply(case_folding)

df["normalisasi"] = df["casefold"].apply(normalize)

df["token"] = df["normalisasi"].apply(tokenize)

df["stopword"] = df["token"].apply(remove_stopwords)

df["stemming"] = df["stopword"].apply(stemming)

df["final_text"] = df["stemming"].apply(
    lambda x: " ".join(x)
)

# =====================================================
# Teacher Sentiment
# ===================================================== 
teacher_sentiment = load_sentiment_teacher()

df["teacher_sentiment"] = df["final_text"].apply(

    lambda x:
    sentiment_map[
        teacher_sentiment(x)[0]["label"]
    ]

)
# =====================================================
# Teacher Emotion
# ===================================================== 
teacher_emotion = load_emotion_teacher()

# =====================================================
# Translate
# ===================================================== 

def translate_text(text):

    try:

        return GoogleTranslator(
            source="id",
            target="en"
        ).translate(text)

    except:

        return text
df["translated"] = df["final_text"].apply(
    translate_text
)

# =====================================================
# Emotion
# ===================================================== 

def predict_emotion(text):

    result = teacher_emotion(text)[0]

    emotion = result["label"]

    return emotion_mapping.get(
        emotion,
        "Frustasi"
    )
def predict_emotion(text):

    result = teacher_emotion(text)[0]

    emotion = result["label"]

    return emotion_mapping.get(
        emotion,
        "Frustasi"
    )

df["teacher_emotion"] = (
    df["translated"]
    .apply(predict_emotion)
)

# =====================================================
# Tokenizer IndoBERT Tab
# ===================================================== 
tokenizer, model = load_indobert()

sample = df["final_text"].iloc[0]

encoded = tokenizer(
    sample,
    truncation=True,
    max_length=128
)
tokenizer.tokenize(sample)

encoded["input_ids"]

encoded["attention_mask"]


# =====================================================
# Embedding Tab
# ===================================================== 

inputs = tokenizer(
    sample,
    return_tensors="pt"
)
with torch.no_grad():

    outputs = model(
        **inputs
    )
	
# =====================================================
# CLS Embedding:
# ===================================================== 
cls_embedding = (
    outputs.last_hidden_state[:,0,:]
)
cls_embedding.shape

# =====================================================
# Visualisasi PCA:
# ===================================================== 
emb = cls_embedding.cpu().numpy()

pca = PCA(
    n_components=2
)

result = pca.fit_transform(
    emb
)
pca_df = pd.DataFrame(
    result,
    columns=["PC1","PC2"]
)
fig = px.scatter(
    pca_df,
    x="PC1",
    y="PC2"
)

st.plotly_chart(fig)

# =====================================================
# Dashboard Sentiment
# ===================================================== 
fig1 = px.pie(
    df,
    names="teacher_sentiment"
)
# =====================================================
# Dashboard Emotion
# ===================================================== 

fig2 = px.bar(
    df["teacher_emotion"]
      .value_counts()
      .reset_index(),
    x="teacher_emotion",
    y="count"
)

# =====================================================
# Crosstab
# =====================================================
with tab7:

    st.plotly_chart(fig1)

    cross = pd.crosstab(
        df["teacher_sentiment"],
        df["teacher_emotion"]
    )

    st.dataframe(cross)

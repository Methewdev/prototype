"""
=====================================================
INDOBERT MODULE
=====================================================
"""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import pandas as pd
import streamlit as st
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel
)

# =====================================================
# LOAD MODEL
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
# TOKENIZER
# =====================================================

def tokenizer_process(text):

    tokenizer, _ = load_indobert()

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128
    )

    return {

        "tokens":
            tokenizer.tokenize(text),

        "input_ids":
            encoded["input_ids"],

        "attention_mask":
            encoded["attention_mask"]

    }

# =====================================================
# EMBEDDING
# =====================================================

# =====================================================
# EMBEDDING DATASET
# =====================================================

def embedding_dataset(df):

    tokenizer, model = load_indobert()

    embeddings = []

    for text in df["final_text"].astype(str):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.no_grad():

            outputs = model(**inputs)

        cls_embedding = (
            outputs.last_hidden_state[:, 0, :]
            .cpu()
            .numpy()[0]
        )

        embeddings.append(cls_embedding)

    return np.array(embeddings)
# =====================================================
# KMEANS CLUSTERING
# =====================================================

def clustering_process(df, n_cluster=3):

    X = embedding_dataset(df)

    kmeans = KMeans(

        n_clusters=n_cluster,

        random_state=42,

        n_init="auto"

    )

    cluster = kmeans.fit_predict(X)

    df["cluster"] = cluster

    cluster_name = {

        0:"Cluster 1",

        1:"Cluster 2",

        2:"Cluster 3"

    }

    df["cluster_name"] = (

        df["cluster"]

        .map(cluster_name)

    )

    score = silhouette_score(

        X,

        cluster

    )

    pca = PCA(

        n_components=2

    )

    pca_result = pca.fit_transform(X)

    df["PCA1"] = pca_result[:,0]

    df["PCA2"] = pca_result[:,1]

    return df, score

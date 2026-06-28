"""
=====================================================
CLUSTERING MODULE
=====================================================
K-Means Clustering using IndoBERT Embedding
=====================================================
"""

import numpy as np
import pandas as pd
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from mdoules.indobert import load_indobert


# =====================================================
# EMBEDDING DATASET
# =====================================================

def embedding_dataset(df):

    tokenizer, model = load_indobert()

    embeddings = []

    model.eval()

    for text in df["final_text"].fillna("").astype(str):

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
# ELBOW METHOD
# =====================================================

def elbow_method(X, max_cluster=8):

    inertia = []

    for k in range(2, max_cluster + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        inertia.append(model.inertia_)

    return inertia


# =====================================================
# KMEANS CLUSTERING
# =====================================================

def clustering_process(df, n_cluster=3):

    X = embedding_dataset(df)

    model = KMeans(
        n_clusters=n_cluster,
        random_state=42,
        n_init=10
    )

    cluster = model.fit_predict(X)

    df = df.copy()

    df["cluster"] = cluster

    df["cluster_name"] = (
        "Cluster "
        + (df["cluster"] + 1).astype(str)
    )

    silhouette = silhouette_score(
        X,
        cluster
    )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    pca_result = pca.fit_transform(X)

    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    return df, silhouette, model


# =====================================================
# CLUSTER SUMMARY
# =====================================================

def cluster_summary(df):

    summary = (

        df.groupby("cluster_name")

        .agg(

            Jumlah_Review=("cluster", "count"),

            Rating_Rata2=("score", "mean"),

            Sentiment_Dominan=("teacher_sentiment", lambda x: x.mode()[0]),

            Emotion_Dominan=("teacher_emotion", lambda x: x.mode()[0])

        )

        .reset_index()

    )

    summary["Rating_Rata2"] = summary["Rating_Rata2"].round(2)

    return summary

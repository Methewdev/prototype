"""
=====================================================
PREPROCESSING MODULE
=====================================================
Tahapan:
1. Cleaning
2. Case Folding
3. Normalization
4. Tokenization
5. Stopword Removal
6. Stemming
=====================================================
"""

import re
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

# =====================================================
# NORMALIZATION DICTIONARY
# =====================================================

NORMALIZATION_DICT = {

    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",

    "yg": "yang",
    "dg": "dengan",
    "dgn": "dengan",
    "dr": "dari",

    "tp": "tapi",
    "krn": "karena",

    "udh": "sudah",
    "sdh": "sudah",

    "bgt": "banget",

    "aja": "saja",

    "trs": "terus",

    "sm": "sama"

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

stemmer = StemmerFactory().create_stemmer()

# =====================================================
# CLEANING
# =====================================================

def cleaning(text):

    text = str(text)

    # URL
    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    # Mention
    text = re.sub(r"@\w+", "", text)

    # Hashtag
    text = re.sub(r"#\w+", "", text)

    # Angka
    text = re.sub(r"\d+", " ", text)

    # Emoji / Simbol
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Multiple Space
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =====================================================
# CASE FOLDING
# =====================================================

def case_folding(text):

    return text.lower()

# =====================================================
# NORMALIZATION
# =====================================================

def normalization(text):

    words = text.split()

    words = [

        NORMALIZATION_DICT.get(
            word,
            word
        )

        for word in words

    ]

    return " ".join(words)

# =====================================================
# TOKENIZATION
# =====================================================

def tokenization(text):

    return text.split()

# =====================================================
# STOPWORD REMOVAL
# =====================================================

def stopword_removal(tokens):

    return [

        word

        for word in tokens

        if word not in stop_words

    ]

# =====================================================
# STEMMING
# =====================================================

def stemming(tokens):

    return [

        stemmer.stem(word)

        for word in tokens

    ]

# =====================================================
# JOIN TOKEN
# =====================================================

def join_tokens(tokens):

    return " ".join(tokens)

# =====================================================
# PIPELINE
# =====================================================

def preprocessing_pipeline(df, review_col):

    process_df = df.copy()

    # Cleaning
    process_df["cleaning"] = (
        process_df[review_col]
        .astype(str)
        .apply(cleaning)
    )

    # Case Folding
    process_df["casefold"] = (
        process_df["cleaning"]
        .apply(case_folding)
    )

    # Normalization
    process_df["normalisasi"] = (
        process_df["casefold"]
        .apply(normalization)
    )

    # Tokenization
    process_df["token"] = (
        process_df["normalisasi"]
        .apply(tokenization)
    )

    # Stopword Removal
    process_df["stopword"] = (
        process_df["token"]
        .apply(stopword_removal)
    )

    # Stemming
    process_df["stemming"] = (
        process_df["stopword"]
        .apply(stemming)
    )

    # Final Text
    process_df["final_text"] = (
        process_df["stemming"]
        .apply(join_tokens)
    )

    return process_df

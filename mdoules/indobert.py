"""
=====================================================
INDOBERT MODULE
=====================================================
"""

import torch
import pandas as pd
import streamlit as st

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

def embedding_process(text):

    tokenizer, model = load_indobert()

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True

    )

    with torch.no_grad():

        outputs = model(**inputs)

    cls = outputs.last_hidden_state[:,0,:]

    embedding = cls.cpu().numpy()

    return pd.DataFrame(
        embedding
    )

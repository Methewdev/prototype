"""
=====================================================
LOADER MODULE
=====================================================
Fungsi:
1. Upload Dataset
2. Membaca CSV / Excel
3. Auto Detect Separator
4. Auto Detect Kolom Review
5. Informasi Dataset
=====================================================
"""

import streamlit as st
import pandas as pd


# ===============================================
# Upload Dataset
# ===============================================

def upload_dataset():

    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Dataset",
        type=["csv", "xlsx"]
    )

    return uploaded_file


# ===============================================
# Load Dataset
# ===============================================

def load_dataset(uploaded_file):

    if uploaded_file is None:
        return None

    try:

        # ==========================
        # Excel
        # ==========================
        if uploaded_file.name.endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        # ==========================
        # CSV
        # ==========================
        else:

            separators = [
                ";",
                ",",
                "\t",
                "|"
            ]

            df = None

            for sep in separators:

                try:

                    uploaded_file.seek(0)

                    temp = pd.read_csv(
                        uploaded_file,
                        sep=sep,
                        on_bad_lines="skip"
                    )

                    if temp.shape[1] > 1:
                        df = temp
                        break

                except:
                    pass

            if df is None:
                raise Exception(
                    "Separator CSV tidak dikenali."
                )

        return df

    except Exception as e:

        st.error(
            f"Gagal membaca dataset : {e}"
        )

        return None


# ===============================================
# Auto Detect Review Column
# ===============================================

def detect_review_column(df):

    candidates = [

        "content",
        "review",
        "reviews",
        "comment",
        "comments",
        "ulasan",
        "text"

    ]

    for col in candidates:

        if col in df.columns:

            return col

    return st.sidebar.selectbox(
        "Pilih Kolom Review",
        df.columns
    )


# ===============================================
# Sidebar Dataset Info
# ===============================================

def show_dataset_info(df):

    st.sidebar.markdown("---")

    st.sidebar.subheader("📊 Informasi Dataset")

    st.sidebar.write(
        f"Jumlah Baris : {len(df)}"
    )

    st.sidebar.write(
        f"Jumlah Kolom : {len(df.columns)}"
    )

    st.sidebar.write(
        f"Missing Value : {df.isnull().sum().sum()}"
    )


# ===============================================
# Preview Dataset
# ===============================================

def preview_dataset(df):

    st.subheader("📄 Preview Dataset")

    st.dataframe(
        df.head(10),
        width="stretch"
    )


# ===============================================
# Data Type
# ===============================================

def dataset_info(df):

    info_df = pd.DataFrame({

        "Column": df.columns,

        "Data Type": df.dtypes.astype(str)

    })

    st.subheader("📋 Informasi Kolom")

    st.dataframe(
        info_df,
        width="stretch"
    )


# ===============================================
# Missing Value
# ===============================================

def missing_value(df):

    miss = pd.DataFrame({

        "Column": df.columns,

        "Missing": df.isnull().sum()

    })

    st.subheader("📌 Missing Value")

    st.dataframe(
        miss,
        width="stretch"
    )


# ===============================================
# Rating Distribution
# ===============================================

def rating_distribution(df):

    import plotly.express as px

    if "score" not in df.columns:
        return

    rating = (

        df["score"]

        .value_counts()

        .sort_index()

        .reset_index()

    )

    rating.columns = [

        "Rating",

        "Jumlah"

    ]

    fig = px.bar(

        rating,

        x="Rating",

        y="Jumlah",

        text="Jumlah",

        title="Distribusi Rating"

    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# =====================================================
# LIVIN NLP ANALYSIS DASHBOARD
# =====================================================

import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# Import mdoules
# =====================================================

from mdoules.loader import (
    upload_dataset,
    load_dataset,
    detect_review_column,
    show_dataset_info
)

from mdoules.preprocessing import (
    preprocessing_pipeline
)

from mdoules.teacher import (
    teacher_pipeline
)

from mdoules.indobert import (
    tokenizer_process,
    embedding_process
)

from mdoules.dashboard import (
    dashboard_metrics,
    sentiment_chart,
    emotion_chart,
    sentiment_vs_emotion,
    sentiment_table,
    emotion_table,
    top_words,
    preview_result,
    download_result
)
from mdoules.scraper import (
    scrape_google_play,
    APP_MAPPING
)
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Dashboard",
    page_icon="📊",
    layout="wide"

)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>
.main{
    padding-top:1rem;
}

.block-container{

    padding-top:2rem;

}

h1,h2,h3{

    color:#0F4C81;

}

</style>

""",unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.title("📊 Analysis Dashboard Sentiment and Emotion")

st.markdown("""
Prototype Analisis Sentiment dan Emotion Review Livin' by Mandiri
menggunakan Teacher Model dan IndoBERT.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.subheader("🌐 Google Play Scraping")

app_name = st.sidebar.selectbox(
    "📱 Pilih Mobile Banking",
    list(APP_MAPPING.keys())
)

total_review = st.sidebar.slider(
    "Jumlah Review",
    100,
    3000,
    500,
    100
)

if st.sidebar.button("🌐 Scraping Review"):

    with st.spinner("Mengambil review..."):

        df = scrape_google_play(
            app_name,
            total_review
        )

    st.session_state.df = df

    st.sidebar.success(
        f"✅ {len(df)} review berhasil diambil."
    )
# =====================================================
# LOAD DATASET
# =====================================================

if "df" not in st.session_state:

    st.info("Silakan lakukan scraping terlebih dahulu.")

    st.stop()

df = st.session_state.df

review_col = "content"
# =====================================================
# REVIEW COLUMN
# =====================================================

review_col = "content"

show_dataset_info(df)
# =====================================================
# SESSION STATE
# =====================================================

if "processed_df" not in st.session_state:

    st.session_state.processed_df = None

# =====================================================
# BUTTON
# =====================================================

run = st.sidebar.button(
    "🚀 Jalankan Analisis"
)

# =====================================================
# PREPROCESSING
# =====================================================

if run:

    with st.spinner("Melakukan preprocessing..."):

        process_df = preprocessing_pipeline(
            df,
            review_col
        )

    with st.spinner("Menjalankan Teacher Model..."):

        process_df = teacher_pipeline(
            process_df
        )

    st.session_state.processed_df = process_df

    st.success(
        "Analisis berhasil dilakukan."
    )

# =====================================================
# TAB
# =====================================================

tab1,\
tab2,\
tab3,\
tab4,\
tab5,\
tab6,\
tab7,\
tab8,\
tab9,\
tab10,\
tab11,\
tab12 = st.tabs([

"📊 Data Understanding",

"🧹 Cleaning",

"🔤 Case Folding",

"📝 Normalization",

"🔪 Tokenization",

"🚫 Stopword Removal",

"🌱 Stemming",

"😊 Teacher Sentiment",

"😡 Teacher Emotion",

"🤖 IndoBERT Tokenizer",

"🔢 IndoBERT Embedding",

"📈 Dashboard"

])

# =====================================================
# TAB 1 : DATA UNDERSTANDING
# =====================================================

with tab1:

    st.header("📊 Data Understanding")

    st.write(
        """
        Tahap Data Understanding bertujuan untuk memahami karakteristik dataset
        sebelum dilakukan preprocessing dan analisis menggunakan model NLP.
        """
    )

    st.markdown("---")

    # =====================================================
    # METRIC
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Review",
        len(df)
    )

    col2.metric(
        "Jumlah Kolom",
        len(df.columns)
    )

    col3.metric(
        "Missing Value",
        int(df.isnull().sum().sum())
    )

    avg_length = (
        df[review_col]
        .astype(str)
        .apply(len)
        .mean()
    )

    col4.metric(
        "Rata-rata Karakter",
        round(avg_length)
    )

    st.markdown("---")

    # =====================================================
    # PREVIEW DATASET
    # =====================================================

    st.subheader("📄 Preview Dataset")

    preview_df = df.copy()
    preview_df.index = range(1, len(preview_df) + 1)

    st.dataframe(
    preview_df,
    width="stretch"
    )
    # =====================================================
    # MISSING VALUE
    # =====================================================

    st.subheader("📌 Missing Value")

    missing_df = pd.DataFrame({

        "Kolom": df.columns,

        "Jumlah Missing": df.isnull().sum()

    })

    st.dataframe(
        missing_df,
        width="stretch"
    )

    st.markdown("---")
    # =====================================================
    # INFORMASI KOLOM
    # =====================================================

    st.subheader("📋 Informasi Dataset")

    info_df = pd.DataFrame({

        "Kolom": df.columns,

        "Tipe Data": df.dtypes.astype(str)

    })

    st.dataframe(
        info_df,
        width="stretch"
    )

    st.markdown("---")

    # =====================================================
    # DISTRIBUSI RATING
    # =====================================================

    if "score" in df.columns:

        st.subheader("⭐ Distribusi Rating")

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

        fig.update_layout(
            xaxis_title="Rating",
            yaxis_title="Jumlah Review"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.info(
            "Kolom 'score' tidak ditemukan."
        )
# =====================================================
# TAB 2 : CLEANING
# =====================================================

with tab2:

    st.header("🧹 Text Cleaning")

    st.write("""
    Tahap **Cleaning** bertujuan membersihkan teks dari karakter
    yang tidak diperlukan sebelum diproses lebih lanjut.

    Karakter yang dihapus meliputi:

    - URL
    - Mention (@username)
    - Hashtag (#)
    - Angka
    - Emoji
    - Simbol
    - Spasi berlebih
    """)

    # ===========================
    # CEK HASIL ANALISIS
    # ===========================

    if st.session_state.processed_df is None:

        st.warning(
            "Silakan klik **Jalankan Analisis** terlebih dahulu."
        )

    else:

        process_df = st.session_state.processed_df

        st.markdown("---")

        st.subheader("📊 Statistik Cleaning")

        total_review = len(process_df)

        total_char_before = (
            process_df[review_col]
            .astype(str)
            .str.len()
            .sum()
        )

        total_char_after = (
            process_df["cleaning"]
            .astype(str)
            .str.len()
            .sum()
        )

        removed_char = total_char_before - total_char_after

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Jumlah Review",
            total_review
        )

        col2.metric(
            "Karakter Sebelum",
            f"{total_char_before:,}"
        )

        col3.metric(
            "Karakter Dihapus",
            f"{removed_char:,}"
        )

        st.markdown("---")

        st.subheader("🔍 Contoh Hasil Cleaning")

        preview = process_df[[
            review_col,
            "cleaning"
        ]].copy()

        preview.columns = [
            "Review Asli",
            "Hasil Cleaning"
        ]
        # Tambahkan nomor urut
        preview.insert(
        0,
        "No",
        range(1, len(preview) + 1)
        )
        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        st.subheader("📄 Detail Perubahan")

        index = st.number_input(
            "Pilih Index Data",
            0,
            len(process_df)-1,
            0
        )

        st.write("### Sebelum")

        st.info(
            process_df.loc[
                index,
                review_col
            ]
        )

        st.write("### Sesudah")

        st.success(
            process_df.loc[
                index,
                "cleaning"
            ]
        )

        st.markdown("---")

        st.subheader("📋 Hasil Cleaning")

        st.dataframe(

            process_df[
                [
                    review_col,
                    "cleaning"
                ]
            ],

            width="stretch"

        )
# =====================================================
# TAB 3 : CASE FOLDING
# =====================================================

with tab3:

    st.header("🔤 Case Folding")

    st.write("""
    Tahap **Case Folding** bertujuan mengubah seluruh huruf menjadi
    huruf kecil (lowercase).

    Contoh:

    **'Aplikasi BAGUS Sekali'**

    menjadi

    **'aplikasi bagus sekali'**
    """)

    # =====================================================
    # Cek hasil preprocessing
    # =====================================================

    if st.session_state.processed_df is None:

        st.warning(
            "Silakan klik **Jalankan Analisis** terlebih dahulu."
        )

    else:

        process_df = st.session_state.processed_df

        st.markdown("---")

        # =====================================================
        # Statistik
        # =====================================================

        st.subheader("📊 Statistik Case Folding")

        total_review = len(process_df)

        total_upper = sum(

            sum(
                1
                for c in str(text)
                if c.isupper()
            )

            for text in process_df["cleaning"]

        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Jumlah Review",
            total_review
        )

        col2.metric(
            "Huruf Kapital",
            total_upper
        )

        st.markdown("---")

        # =====================================================
        # Contoh Hasil
        # =====================================================

        st.subheader("🔍 Contoh Case Folding")

        preview = process_df[
            [
                "cleaning",
                "casefold"
            ]
        ].copy()

        preview.columns = [

            "Sebelum",

            "Sesudah"

        ]
        # Tambahkan nomor urut
        preview.insert(
        0,
        "No",
        range(1, len(preview) + 1)
        )
        st.dataframe(

            preview,
            width="stretch",
            hide_index=True

        )

        st.markdown("---")

        # =====================================================
        # Detail Perubahan
        # =====================================================

        st.subheader("📄 Detail Perubahan")

        index = st.number_input(

            "Pilih Index",

            min_value=0,

            max_value=len(process_df)-1,

            value=0,

            key="casefold_index"

        )

        st.write("### Sebelum")

        st.info(

            process_df.loc[
                index,
                "cleaning"
            ]

        )

        st.write("### Sesudah")

        st.success(

            process_df.loc[
                index,
                "casefold"
            ]

        )

        st.markdown("---")

        # =====================================================
        # Tabel
        # =====================================================

        st.subheader("📋 Hasil Case Folding")

        st.dataframe(

            process_df[
                [
                    "cleaning",
                    "casefold"
                ]
            ],

            width="stretch"

        )
# =====================================================
# TAB 4 : NORMALIZATION
# =====================================================

with tab4:

    st.header("📝 Normalization")

    st.write("""
    Tahap Normalization mengubah kata tidak baku menjadi kata baku.
    Contoh:

    - gk → tidak
    - yg → yang
    - bgt → banget
    - tp → tapi
    """)

    if st.session_state.processed_df is None:

        st.warning("Silakan klik Jalankan Analisis terlebih dahulu.")

    else:

        process_df = st.session_state.processed_df

        st.subheader("📊 Statistik")

        col1, col2 = st.columns(2)

        col1.metric(
            "Jumlah Review",
            len(process_df)
        )

        col2.metric(
            "Kolom",
            "normalisasi"
        )

        st.markdown("---")

        preview = process_df[
            [
                "casefold",
                "normalisasi"
            ]
        ].copy()

        preview.columns = [
            "Sebelum",
            "Sesudah"
        ]
        # Tambahkan nomor urut
        preview.insert(
             0,
            "No",
            range(1, len(preview) + 1)
        )
        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        idx = st.number_input(
            "Pilih Index",
            0,
            len(process_df)-1,
            0,
            key="norm"
        )

        st.info(
            process_df.loc[idx,"casefold"]
        )

        st.success(
            process_df.loc[idx,"normalisasi"]
        )

        st.markdown("---")

        st.dataframe(
            process_df[
                [
                    "casefold",
                    "normalisasi"
                ]
            ],
            width="stretch"
        )
# =====================================================
# TAB 5 : TOKENIZATION
# =====================================================

with tab5:

    st.header("🔪 Tokenization")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik Jalankan Analisis.")

    else:

        process_df = st.session_state.processed_df

        preview = process_df[
            [
                "normalisasi",
                "token"
            ]
        ].copy()
        # Tambahkan nomor urut
        preview.insert(
            0,
            "No",
            range(1, len(preview) + 1)
        )
        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )

        idx = st.number_input(
            "Index",
            0,
            len(process_df)-1,
            0,
            key="token"
        )

        st.info(
            process_df.loc[idx,"normalisasi"]
        )

        st.success(
            process_df.loc[idx,"token"]
        )
# =====================================================
# TAB 6 : STOPWORD REMOVAL
# =====================================================

with tab6:

    st.header("🚫 Stopword Removal")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik **Jalankan Analisis** terlebih dahulu.")

    else:

        process_df = st.session_state.processed_df

        # ==========================================
        # Membuat DataFrame Preview
        # ==========================================

        preview = process_df[
            [
                "token",
                "stopword"
            ]
        ].copy()

        preview.columns = [
            "Sebelum Stopword",
            "Sesudah Stopword"
        ]

        # Tambahkan nomor urut
        preview.insert(
            0,
            "No",
            range(1, len(preview) + 1)
        )

        st.subheader("📋 Hasil Stopword Removal")

        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        # ==========================================
        # Detail Perubahan
        # ==========================================

        st.subheader("📄 Detail Perubahan")

        idx = st.number_input(
            "Pilih Index",
            min_value=0,
            max_value=len(process_df) - 1,
            value=0,
            key="stop"
        )

        st.write("### Sebelum Stopword Removal")

        st.info(
            process_df.loc[idx, "token"]
        )

        st.write("### Sesudah Stopword Removal")

        st.success(
            process_df.loc[idx, "stopword"]
        )
# =====================================================
# TAB 7 : STEMMING
# =====================================================

with tab7:

    st.header("🌱 Stemming")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik **Jalankan Analisis** terlebih dahulu.")

    else:

        process_df = st.session_state.processed_df

        # ==========================================
        # Membuat DataFrame Preview
        # ==========================================

        preview = process_df[
            [
                "stopword",
                "stemming"
            ]
        ].copy()

        preview.columns = [
            "Sebelum Stemming",
            "Sesudah Stemming"
        ]

        # Tambahkan nomor urut
        preview.insert(
            0,
            "No",
            range(1, len(preview) + 1)
        )

        st.subheader("📋 Hasil Stemming")

        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )

        st.markdown("---")

        # ==========================================
        # Detail Perubahan
        # ==========================================

        st.subheader("📄 Detail Perubahan")

        idx = st.number_input(
            "Pilih Index",
            min_value=0,
            max_value=len(process_df) - 1,
            value=0,
            key="stem"
        )

        st.write("### Sebelum Stemming")

        st.info(
            process_df.loc[idx, "stopword"]
        )

        st.write("### Sesudah Stemming")

        st.success(
            process_df.loc[idx, "stemming"]
        )
# =====================================================
# TAB 8 Teacher Sentiment
# =====================================================

with tab8:

    st.header("😊 Teacher Sentiment")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik Jalankan Analisis.")

    else:

        process_df = st.session_state.processed_df

        preview = process_df[
           [
                "final_text",
                "teacher_sentiment",
                "sentiment_score"
            ]
        ].copy()

        preview.insert(
            0,
            "No",
            range(1, len(preview) + 1)
        )

        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )
# =====================================================
# TAB 9 Teacher Emotion
# =====================================================

with tab9:

    st.header("😡 Teacher Emotion")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik Jalankan Analisis.")

    else:

        process_df = st.session_state.processed_df

        preview = process_df[
            [
                "final_text",
                "teacher_sentiment",
                "sentiment_score"
            ]
        ].copy()

        preview.insert(
            0,
            "No",
            range(1, len(preview) + 1)
        )

        st.dataframe(
            preview,
            width="stretch",
            hide_index=True
        )
# =====================================================
# TAB 10 : INDOBERT TOKENIZER
# =====================================================

with tab10:

    st.header("🤖 IndoBERT Tokenizer")

    st.write("""
Tahap ini mengubah hasil preprocessing menjadi token yang dapat dipahami oleh model IndoBERT.

Pipeline:

Review
↓
Cleaning
↓
Case Folding
↓
Normalization
↓
Tokenization
↓
Stopword Removal
↓
Stemming
↓
Tokenizer IndoBERT
""")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik Jalankan Analisis terlebih dahulu.")

    else:

        process_df = st.session_state.processed_df

        idx = st.number_input(
            "Pilih Data",
            0,
            len(process_df)-1,
            0,
            key="bert_token"
        )

        text = process_df.loc[idx,"final_text"]

        st.subheader("Final Text")

        st.code(text)

        result = tokenizer_process(text)

        st.markdown("---")

        st.subheader("Tokens")

        st.write(result["tokens"])

        st.markdown("---")

        st.subheader("Input IDs")

        st.write(result["input_ids"])

        st.markdown("---")

        st.subheader("Attention Mask")

        st.write(result["attention_mask"])

        st.markdown("---")

        st.metric(
            "Jumlah Token",
            len(result["tokens"])
        )
# =====================================================
# TAB 11 : INDOBERT EMBEDDING
# =====================================================

with tab11:

    st.header("🔢 IndoBERT Embedding")

    st.write("""
Embedding merupakan representasi numerik dari setiap review
yang dihasilkan oleh model IndoBERT.

Embedding inilah yang menjadi input model Machine Learning.
""")

    if st.session_state.processed_df is None:

        st.warning("Silakan klik Jalankan Analisis.")

    else:

        process_df = st.session_state.processed_df

        idx = st.number_input(
            "Pilih Data",
            0,
            len(process_df)-1,
            0,
            key="embedding"
        )

        text = process_df.loc[idx,"final_text"]

        embedding = embedding_process(text)

        st.subheader("Embedding Shape")

        st.write(embedding.shape)

        st.markdown("---")

        st.subheader("Embedding Matrix")

        st.dataframe(
            embedding,
            width="stretch"
        )

        st.markdown("---")

        st.metric(
            "Dimensi Embedding",
            embedding.shape[1]
        )
# =====================================================
# TAB 12 : DASHBOARD
# =====================================================

with tab12:

    st.header("📈 Dashboard Analytics")

    if st.session_state.processed_df is None:

        st.warning(
            "Silakan klik **Jalankan Analisis** terlebih dahulu."
        )

    else:

        process_df = st.session_state.processed_df

        dashboard_metrics(process_df)

        st.markdown("---")

        sentiment_chart(process_df)

        st.markdown("---")

        emotion_chart(process_df)

        st.markdown("---")

        sentiment_vs_emotion(process_df)

        st.markdown("---")

        sentiment_table(process_df)

        st.markdown("---")

        emotion_table(process_df)

        st.markdown("---")

        top_words(process_df)

        st.markdown("---")

        preview_result(process_df)

        st.markdown("---")

        download_result(process_df)

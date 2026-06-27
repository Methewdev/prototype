# =====================================================
# LOAD DATASET
# =====================================================

df = None

if uploaded_file is not None:

    try:

        # Excel
        if uploaded_file.name.endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        # CSV
        else:

            try:

                df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    on_bad_lines="skip"
                )

            except:

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    sep=",",
                    on_bad_lines="skip"
                )

        st.sidebar.success(
            f"✅ Dataset berhasil dimuat ({len(df)} review)"
        )

    except Exception as e:

        st.error(
            f"Gagal membaca file: {e}"
        )

        st.stop()

else:

    st.info(
        "📁 Silakan upload dataset terlebih dahulu."
    )

    st.stop()

# =====================================================
# AUTO DETECT REVIEW COLUMN
# =====================================================

review_candidates = [
    "content",
    "review",
    "comment",
    "ulasan",
    "text"
]

review_col = None

for col in review_candidates:

    if col in df.columns:

        review_col = col
        break

# fallback manual
if review_col is None:

    review_col = st.sidebar.selectbox(
        "Pilih Kolom Review",
        df.columns
    )

else:

    st.sidebar.success(
        f"📝 Kolom Review : {review_col}"
    )

# =====================================================
# DATASET INFO
# =====================================================

with st.sidebar:

    st.markdown("---")

    st.subheader("📌 Informasi Dataset")

    st.write(f"Baris : {len(df)}")
    st.write(f"Kolom : {len(df.columns)}")

# =====================================================
# TABS
# =====================================================

tab1 = st.tabs([
    "📊 Data Understanding"
])[0]

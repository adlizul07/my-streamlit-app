import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Media Cleaning & NLP Pipeline", layout="wide")
DetectorFactory.seed = 0

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

# ==========================================
# MODEL
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# ==========================================
# HELPERS
# ==========================================
def clean_text(text):
    if pd.isnull(text):
        return text
    return str(text).replace('_x000D_', ' ').replace('\r', ' ').replace('\n', ' ').strip()

def create_combined_column(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, excluded):
    subset = [c for c in df.columns if c not in excluded]
    return df.drop_duplicates(subset=subset)

# ==========================
# SENTENCE MATCHING
# ==========================
def extract_sentences(text, keywords):
    if pd.isnull(text):
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return "\n".join(
        s for s in sentences
        if any(k.lower() in s.lower() for k in keywords)
    )

def extract_irrelevant(text, keywords):
    if pd.isnull(text):
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return "\n".join(
        s for s in sentences
        if any(k.lower() in s.lower() for k in keywords)
    )

def keyword_match(df, keyword_df, col_name):
    keywords = keyword_df.iloc[:, 0].dropna().astype(str).tolist()
    df[col_name] = df["Combined"].apply(lambda x: extract_sentences(x, keywords))
    return df

def irrelevant_match(df, keyword_df, col_name):
    keywords = keyword_df.iloc[:, 0].dropna().astype(str).tolist()
    df[col_name] = df["Combined"].apply(lambda x: extract_irrelevant(x, keywords))
    return df

# ==========================
# TRANSLATION
# ==========================
def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    def tr(x):
        try:
            return translator.translate(str(x)[:2000])
        except:
            return x

    df["Translated"] = df["Combined"].apply(tr)
    df["Translated"] = df["Translated"].apply(clean_text)
    return df

# ==========================
# CLUSTERING
# ==========================
def cluster(df, threshold):
    model = load_embedding_model()

    texts = df["Combined"].fillna("").astype(str).tolist()
    emb = model.encode(texts, convert_to_numpy=True, batch_size=32)
    emb = normalize(emb)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    df["Cluster"] = clustering.fit_predict(emb)
    return df

# ==========================
# EXCEL EXPORT
# ==========================
def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    buffer.seek(0)
    return buffer

# ==========================================
# UI
# ==========================================
st.title("📊 Data Cleaning & NLP Pipeline")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

# ==========================
# LOAD DATA (CRITICAL FIX)
# ==========================
if file:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    st.session_state.data = pd.read_excel(file, sheet_name=sheet)

# ALWAYS USE SESSION DATA
df = st.session_state.data

if df is not None:
    st.subheader("Original Data Preview")
    st.dataframe(df.head(5))

    # ==========================================
    # STEP 1 — COMBINED
    # ==========================================
    st.header("Step 1 — Create Combined Column")

    cols = st.multiselect("Select columns", df.columns)

    c1, c2 = st.columns(2)

    with c1:
        run = st.button("▶ Run Combined")
    with c2:
        skip = st.button("⏭ Skip Combined")

    if run:
        if len(cols) == 0:
            st.error("❌ Select at least one column")
        else:
            df = create_combined_column(df, cols)
            st.session_state.data = df
            st.success("✅ Combined column created")
            st.dataframe(df.head(5))

    if skip:
        st.info("⏭ Combined step skipped")

    # ==========================================
    # STEP 2 — DUPLICATES
    # ==========================================
    st.header("Step 2 — Remove Duplicates")

    exclude = st.multiselect("Exclude columns", df.columns)

    c1, c2 = st.columns(2)

    with c1:
        run = st.button("▶ Run Duplicates")
    with c2:
        skip = st.button("⏭ Skip Duplicates")

    if run:
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        st.success("✅ Duplicates removed")
        st.dataframe(df.head(5))

    if skip:
        st.info("⏭ Duplicate step skipped")

    # ==========================================
    # STEP 3 — KEYWORDS + IRRELEVANT
    # ==========================================
    st.header("Step 3 — Keyword & Irrelevant Matching")

    kw_file = st.file_uploader("Keyword File", type=["xlsx"])
    irr_file = st.file_uploader("Irrelevant Keyword File", type=["xlsx"])

    kw_col = st.text_input("Keyword Output Column", "Keyword Match")
    irr_col = st.text_input("Irrelevant Output Column", "Irrelevant Match")

    c1, c2 = st.columns(2)

    with c1:
        run_kw = st.button("▶ Run Keyword Match")
    with c2:
        skip_kw = st.button("⏭ Skip Step 3")

    if run_kw:
        if "Combined" not in df.columns:
            st.error("❌ Create Combined column first")
        elif kw_file is None:
            st.error("❌ Upload keyword file")
        else:
            kw_df = pd.read_excel(kw_file)
            df = keyword_match(df, kw_df, kw_col)
            st.session_state.data = df
            st.success("✅ Keyword matching done")
            st.dataframe(df.head(5))

    if irr_file is not None:
        if st.button("▶ Run Irrelevant Match"):
            if "Combined" not in df.columns:
                st.error("❌ Create Combined column first")
            else:
                irr_df = pd.read_excel(irr_file)
                df = irrelevant_match(df, irr_df, irr_col)
                st.session_state.data = df
                st.success("✅ Irrelevant matching done")
                st.dataframe(df.head(5))

    if skip_kw:
        st.info("⏭ Keyword step skipped")

    # ==========================================
    # STEP 4 — TRANSLATION
    # ==========================================
    st.header("Step 4 — Translation")

    c1, c2 = st.columns(2)

    with c1:
        run = st.button("▶ Run Translation")
    with c2:
        skip = st.button("⏭ Skip Translation")

    if run:
        if "Combined" not in df.columns:
            st.error("❌ Combined column missing")
        else:
            df = translate(df)
            st.session_state.data = df
            st.success("✅ Translation completed")
            st.dataframe(df.head(5))

    if skip:
        st.info("⏭ Translation skipped")

    # ==========================================
    # STEP 5 — CLUSTERING
    # ==========================================
    st.header("Step 5 — Clustering")

    thr = st.slider("Cluster strictness", 0.1, 1.0, 0.28)

    c1, c2 = st.columns(2)

    with c1:
        run = st.button("▶ Run Clustering")
    with c2:
        skip = st.button("⏭ Skip Clustering")

    if run:
        if "Combined" not in df.columns:
            st.error("❌ Combined column missing")
        else:
            df = cluster(df, thr)
            st.session_state.data = df
            st.success("✅ Clustering completed")
            st.dataframe(df.head(5))

    if skip:
        st.info("⏭ Clustering skipped")

    # ==========================================
    # FINAL OUTPUT
    # ==========================================
    st.header("Final Output")

    st.dataframe(st.session_state.data.head(5))

    out = to_excel(st.session_state.data)

    st.download_button(
        "📥 Download Excel",
        data=out,
        file_name="output.xlsx"
    )
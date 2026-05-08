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
from openpyxl import load_workbook

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
    text = str(text).replace('_x000D_', ' ').replace('\r', ' ').replace('\n', ' ')
    return text.strip()

def detect_language(text):
    try:
        return detect(str(text))
    except:
        return "unknown"

def create_combined_column(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, excluded):
    subset = [c for c in df.columns if c not in excluded]
    return df.drop_duplicates(subset=subset)

def extract_sentences(text, keywords):
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

file = st.file_uploader("Upload Excel", type=["xlsx"])

if file:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Sheet", xls.sheet_names)
    df = pd.read_excel(file, sheet_name=sheet)

    st.dataframe(df.head())

    # Combined
    cols = st.multiselect("Columns for Combined", df.columns)

    if st.button("Create Combined"):
        df = create_combined_column(df, cols)
        st.session_state.data = df

    # Duplicates
    exclude = st.multiselect("Exclude columns (duplicates)", df.columns)

    if st.button("Remove Duplicates"):
        df = remove_duplicates(df, exclude)
        st.session_state.data = df

    # Keyword
    kw = st.file_uploader("Keyword file", type=["xlsx"], key="kw")

    if kw and st.button("Run Keyword"):
        kw_df = pd.read_excel(kw)
        df = keyword_match(df, kw_df, "Keyword Match")
        st.session_state.data = df

    # Translation
    if st.button("Translate"):
        df = translate(df)
        st.session_state.data = df

    # Clustering
    thr = st.slider("Cluster strictness", 0.1, 1.0, 0.28)

    if st.button("Cluster"):
        df = cluster(df, thr)
        st.session_state.data = df

    st.dataframe(st.session_state.data)

    out = to_excel(st.session_state.data)

    st.download_button(
        "Download Excel",
        data=out,
        file_name="output.xlsx"
    )
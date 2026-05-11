import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from transformers import pipeline
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Data Cleaner Pro",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# ==========================================
# STATUS SYSTEM
# ==========================================
def set_status(step, value):
    st.session_state[f"status_{step}"] = value

def get_status(step):
    return st.session_state.get(f"status_{step}", "Not Run")

def status_icon(status):
    return {
        "Not Run": "🟡",
        "Running": "🔵",
        "Done": "🟢",
        "Error": "🔴",
        "Skipped": "⚪"
    }.get(status, "🟡")

# ==========================================
# MODELS
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ==========================================
# HELPERS
# ==========================================
def clean_text(text):
    if pd.isnull(text):
        return text

    text = str(text)

    # preserve URLs + text integrity
    text = text.replace('_x000D_', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')

    return text.strip()

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(object).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, exclude):
    subset = [c for c in df.columns if c not in exclude]
    return df.drop_duplicates(subset=subset)

def extract_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return " ".join([s for s in sentences if any(k.lower() in s.lower() for k in keywords)])

def generate_cluster_summary(df):
    cluster_map = {}

    for c in df["Cluster"].unique():
        sample = df[df["Cluster"] == c]["Combined"].dropna().astype(str)

        if len(sample) == 0:
            cluster_map[c] = "Empty cluster"
        else:
            cluster_map[c] = " | ".join(sample.head(3).tolist())

    df["Cluster_Description"] = df["Cluster"].map(cluster_map)
    return df

def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    def tr(x):
        try:
            return translator.translate(str(x)[:2000])
        except:
            return x

    df["Translated"] = df["Combined"].apply(tr)
    return df

# ==========================================
# EXCEL EXPORT (FIXED - PRESERVES LINKS)
# ==========================================
def to_excel(df):
    wb = Workbook()
    ws = wb.active

    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)

    # convert URLs into clickable hyperlinks
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                if cell.value.startswith("http"):
                    cell.hyperlink = cell.value
                    cell.style = "Hyperlink"

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📁 Navigation")

file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

output_name = st.sidebar.text_input(
    "📄 Output file name",
    value="output.xlsx"
)

# ==========================================
# LOAD DATA
# ==========================================
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.sidebar.selectbox("Sheet", xls.sheet_names)
    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True

df = st.session_state.data

if df is None:
    st.title("📊 Data Cleaner Pro")
    st.info("Upload file to start")
    st.stop()

# ==========================================
# HEADER
# ==========================================
st.title("📊 Data Cleaner Pro")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 1 — COMBINE
# ==========================================
st.header("🧩 Step 1 — Combine Columns")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Combine"):

    set_status("Combine", "Running")

    try:
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("Combine", "Done")
    except:
        set_status("Combine", "Error")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 2 — DEDUP
# ==========================================
st.header("🧹 Step 2 — Remove Duplicates")

exclude = st.multiselect("Exclude columns", df.columns)

if st.button("▶ Run Deduplication"):

    set_status("Dedup", "Running")

    try:
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        set_status("Dedup", "Done")
    except:
        set_status("Dedup", "Error")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 3 — KEYWORDS
# ==========================================
st.header("🔑 Step 3 — Keyword Matching")

num = st.number_input("Groups", 1, 10, 1)

for i in range(num):

    st.subheader(f"Group {i+1}")

    kw_text = st.text_input("Keywords", key=f"kw{i}")
    tag_col = st.text_input("Tag column", f"Tags_{i+1}")

    extract_sent = st.checkbox("Extract sentences?", key=f"sent{i}")

    sent_col = st.text_input("Sentence column", f"Sent_{i+1}")

    if st.button(f"▶ Run Group {i+1}"):

        set_status("Keyword", "Running")

        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

        df[tag_col] = df["Combined"].apply(
            lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
        )

        if extract_sent:
            df[sent_col] = df["Combined"].apply(
                lambda x: extract_sentences(x, keywords)
            )

        st.session_state.data = df
        set_status("Keyword", "Done")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 4 — TRANSLATION
# ==========================================
st.header("🌍 Step 4 — Translation")

if st.button("▶ Run Translation"):

    set_status("Translate", "Running")

    try:
        df = translate(df)
        st.session_state.data = df
        set_status("Translate", "Done")
    except:
        set_status("Translate", "Error")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 5 — SENTIMENT
# ==========================================
st.header("💬 Step 5 — Sentiment")

source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])
brand_col = st.selectbox("Brand column", df.columns)

if st.button("▶ Run Sentiment"):

    set_status("Sentiment", "Running")

    model = load_sentiment_model()

    results = []

    for _, row in df.iterrows():

        text = str(row[source])
        brand = str(row[brand_col])

        if brand.lower() not in text.lower():
            results.append("NO_MENTION")
            continue

        results.append(model(text[:512])[0]["label"])

    df["Sentiment"] = results
    st.session_state.data = df

    set_status("Sentiment", "Done")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 6 — CLUSTERING
# ==========================================
st.header("📦 Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

if st.button("▶ Run Clustering"):

    set_status("Cluster", "Running")

    model = load_model()

    emb = model.encode(df["Combined"].astype(str).tolist(), convert_to_numpy=True)
    emb = normalize(emb)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    df["Cluster"] = clustering.fit_predict(emb)
    df = generate_cluster_summary(df)

    st.session_state.data = df

    set_status("Cluster", "Done")

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# DOWNLOAD
# ==========================================
st.sidebar.markdown("---")

st.sidebar.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name=output_name
)
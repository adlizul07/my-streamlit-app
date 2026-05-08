import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Data Cleaning & NLP Pipeline", layout="wide")
DetectorFactory.seed = 0

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# ==========================================
# MODEL
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# ==========================================
# HELPERS
# ==========================================
def clean_text(text):
    if pd.isnull(text):
        return text
    return str(text).replace('_x000D_', ' ').replace('\r', ' ').replace('\n', ' ').strip()

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, exclude):
    subset = [c for c in df.columns if c not in exclude]

    if len(subset) == 0:
        return df

    return df.drop_duplicates(subset=subset)

def extract_sentences(text, keywords):
    if pd.isnull(text):
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return "\n".join(
        s for s in sentences
        if any(k.lower() in s.lower() for k in keywords)
    )

def keyword_match(df, kw_df, col):
    keywords = kw_df.iloc[:, 0].dropna().astype(str).tolist()
    df[col] = df["Combined"].apply(lambda x: extract_sentences(x, keywords))
    return df

def irrelevant_match(df, irr_df, col):
    keywords = irr_df.iloc[:, 0].dropna().astype(str).tolist()
    df[col] = df["Combined"].apply(lambda x: extract_sentences(x, keywords))
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

# ==========================================
# CLUSTERING + DESCRIPTION
# ==========================================
def cluster(df, threshold):
    model = load_model()

    texts = df["Combined"].fillna("").astype(str).tolist()
    emb = model.encode(texts, convert_to_numpy=True, batch_size=32)
    emb = normalize(emb)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    labels = clustering.fit_predict(emb)
    df["Cluster"] = labels

    # ===============================
    # CLUSTER DESCRIPTION (YOUR LOGIC)
    # ===============================
    cluster_desc = {}

    for cluster_id in np.unique(labels):
        sample_text = df.loc[df["Cluster"] == cluster_id, "Combined"].iloc[0]
        cluster_desc[cluster_id] = str(sample_text)[:120] + "..."

    df["Cluster Description"] = df["Cluster"].map(cluster_desc)

    return df

# ==========================================
# EXPORT
# ==========================================
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

# ==========================================
# LOAD FILE ONLY ONCE
# ==========================================
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names, key="sheet_select")

    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True

df = st.session_state.data

if df is None:
    st.info("⬆️ Upload file to start")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(5))

# ==========================================
# STEP 1 — COMBINED
# ==========================================
st.header("Step 1 — Combined Column")

cols = st.multiselect("Select columns", df.columns, key="cols")

c1, c2 = st.columns(2)
with c1:
    run = st.button("▶ Run Combined", key="run_combined")
with c2:
    skip = st.button("⏭ Skip Combined", key="skip_combined")

if run:
    if len(cols) == 0:
        st.error("❌ Select columns")
    else:
        df = st.session_state.data
        df = create_combined(df, cols)
        st.session_state.data = df
        st.success("✅ Combined created")
        st.dataframe(df.head(5))

if skip:
    st.info("⏭ Skipped")

# ==========================================
# STEP 2 — DUPLICATES
# ==========================================
st.header("Step 2 — Remove Duplicates")

exclude = st.multiselect("Exclude columns", df.columns, key="exclude")

c1, c2 = st.columns(2)
with c1:
    run = st.button("▶ Run Duplicates", key="run_dup")
with c2:
    skip = st.button("⏭ Skip Duplicates", key="skip_dup")

if run:
    df = st.session_state.data
    df = remove_duplicates(df, exclude)
    st.session_state.data = df
    st.success("✅ Duplicates removed")
    st.dataframe(df.head(5))

if skip:
    st.info("⏭ Skipped")

# ==========================================
# STEP 3 — KEYWORDS
# ==========================================
st.header("Step 3 — Keyword Matching")

kw_file = st.file_uploader("Keyword File", type=["xlsx"], key="kw")
kw_col = st.text_input("Keyword Column", "Keyword Match", key="kwcol")

c1, c2 = st.columns(2)
with c1:
    run = st.button("▶ Run Keyword", key="run_kw")
with c2:
    skip = st.button("⏭ Skip Keyword", key="skip_kw")

if run:
    df = st.session_state.data

    if "Combined" not in df.columns:
        st.error("❌ Combined missing")
    elif kw_file:
        kw_df = pd.read_excel(kw_file)
        df = keyword_match(df, kw_df, kw_col)
        st.session_state.data = df
        st.success("✅ Keyword done")
        st.dataframe(df.head(5))

if skip:
    st.info("⏭ Skipped")

# ==========================================
# STEP 4 — IRRELEVANT
# ==========================================
st.header("Step 4 — Irrelevant Matching")

irr_file = st.file_uploader("Irrelevant File", type=["xlsx"], key="irr")
irr_col = st.text_input("Irrelevant Column", "Irrelevant Match", key="irrcol")

c1, c2 = st.columns(2)
with c1:
    run = st.button("▶ Run Irrelevant", key="run_irr")
with c2:
    skip = st.button("⏭ Skip Irrelevant", key="skip_irr")

if run:
    df = st.session_state.data

    if "Combined" not in df.columns:
        st.error("❌ Combined missing")
    elif irr_file:
        irr_df = pd.read_excel(irr_file)
        df = irrelevant_match(df, irr_df, irr_col)
        st.session_state.data = df
        st.success("✅ Irrelevant done")
        st.dataframe(df.head(5))

if skip:
    st.info("⏭ Skipped")

# ==========================================
# STEP 5 — TRANSLATION
# ==========================================
st.header("Step 5 — Translation")

c1, c2 = st.columns(2)
with c1:
    run = st.button("▶ Run Translate", key="run_tr")
with c2:
    skip = st.button("⏭ Skip Translate", key="skip_tr")

if run:
    df = st.session_state.data

    if "Combined" not in df.columns:
        st.error("❌ Combined missing")
    else:
        df = translate(df)
        st.session_state.data = df
        st.success("✅ Translation done")
        st.dataframe(df.head(5))

if skip:
    st.info("⏭ Skipped")

# ==========================================
# STEP 6 — CLUSTERING
# ==========================================
st.header("Step 6 — Clustering")

threshold = st.slider("Strictness", 0.1, 1.0, 0.28, key="thr")

c1, c2 = st.columns(2)
with c1:
    run = st.button("▶ Run Cluster", key="run_cluster")
with c2:
    skip = st.button("⏭ Skip Cluster", key="skip_cluster")

if run:
    df = st.session_state.data

    if "Combined" not in df.columns:
        st.error("❌ Combined missing")
    else:
        df = cluster(df, threshold)
        st.session_state.data = df
        st.success("✅ Clustering done")
        st.dataframe(df.head(5))

if skip:
    st.info("⏭ Skipped")

# ==========================================
# FINAL OUTPUT
# ==========================================
st.header("Final Output")

st.dataframe(st.session_state.data.head(5))

excel = to_excel(st.session_state.data)

st.download_button(
    "📥 Download Excel",
    data=excel,
    file_name="output.xlsx"
)
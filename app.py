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

def parse_keywords(text):
    if not text:
        return []
    return [k.strip() for k in text.split(",") if k.strip()]

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

    cluster_desc = {}

    for c in np.unique(labels):
        sample = df.loc[df["Cluster"] == c, "Combined"].iloc[0]
        cluster_desc[c] = str(sample)[:120] + "..."

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

# LOAD FILE
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

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
st.info("Combine selected columns into one 'Combined' text column for NLP processing.")

cols = st.multiselect("Select columns to combine", df.columns)

if st.button("▶ Run Combined"):
    if len(cols) == 0:
        st.error("❌ Please select columns")
    else:
        df = create_combined(df, cols)
        st.session_state.data = df
        st.success("✅ Combined column created")
        st.dataframe(df.head(5))

if st.button("⏭ Skip Combined"):
    st.info("Skipped Combined step")

# ==========================================
# STEP 2 — DUPLICATES
# ==========================================
st.header("Step 2 — Remove Duplicates")
st.info("Remove duplicate rows based on selected columns.")

exclude = st.multiselect("Exclude columns from duplicate check", df.columns)

if st.button("▶ Run Duplicates"):
    df = remove_duplicates(df, exclude)
    st.session_state.data = df
    st.success("✅ Duplicates removed")
    st.dataframe(df.head(5))

if st.button("⏭ Skip Duplicates"):
    st.info("Skipped Duplicates step")

# ==========================================
# STEP 3 — MULTI KEYWORD MATCHING
# ==========================================
st.header("Step 3 — Keyword Matching (Multi Groups)")
st.info("Extract sentences that contain keywords from Combined column. You can upload Excel OR type keywords manually (use commas for multiple keywords).")

num_groups = st.number_input("How many keyword groups?", 1, 10, 1)

for i in range(num_groups):

    st.markdown(f"### 🔹 Keyword Group {i+1}")

    kw_file = st.file_uploader(
        f"Upload keyword file (Group {i+1})",
        type=["xlsx"],
        key=f"kw_file_{i}"
    )

    kw_text = st.text_input(
        f"OR type keywords (comma separated) — Group {i+1}",
        key=f"kw_text_{i}"
    )

    kw_col = st.text_input(
        f"Output column name — Group {i+1}",
        value=f"Keyword_Group_{i+1}",
        key=f"kw_col_{i}"
    )

    if st.button(f"▶ Run Group {i+1}", key=f"run_kw_{i}"):

        df = st.session_state.data
        keywords = []

        if kw_file:
            kw_df = pd.read_excel(kw_file)
            keywords = kw_df.iloc[:, 0].dropna().astype(str).tolist()

        elif kw_text:
            keywords = parse_keywords(kw_text)

        else:
            st.error("❌ Please upload file OR type keywords (comma separated)")
            st.stop()

        df[kw_col] = df["Combined"].apply(
            lambda x: extract_sentences(x, keywords)
        )

        st.session_state.data = df
        st.success(f"✅ Keyword Group {i+1} completed")
        st.dataframe(df.head(5))

# ==========================================
# STEP 4 — TRANSLATION
# ==========================================
st.header("Step 4 — Translation")
st.info("Translate Combined column into English for standardization.")

if st.button("▶ Run Translation"):
    df = st.session_state.data

    if "Combined" not in df.columns:
        st.error("❌ Combined column missing")
    else:
        df = translate(df)
        st.session_state.data = df
        st.success("✅ Translation completed")
        st.dataframe(df.head(5))

if st.button("⏭ Skip Translation"):
    st.info("Skipped Translation step")

# ==========================================
# STEP 5 — CLUSTERING
# ==========================================
st.header("Step 5 — Clustering")
st.info("Group similar content together using AI embeddings. Lower strictness = more clusters.")

threshold = st.slider("Clustering Strictness", 0.1, 1.0, 0.28)

if st.button("▶ Run Clustering"):
    df = st.session_state.data

    if "Combined" not in df.columns:
        st.error("❌ Combined column missing")
    else:
        df = cluster(df, threshold)
        st.session_state.data = df
        st.success("✅ Clustering completed")
        st.dataframe(df.head(5))

if st.button("⏭ Skip Clustering"):
    st.info("Skipped Clustering step")

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
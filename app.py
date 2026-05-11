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
from transformers import pipeline

# ==========================================
# CONFIG + THEME
# ==========================================
st.set_page_config(
    page_title="Data Cleaner Pro",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
.block-container {padding: 2rem 2rem 3rem 2rem;}
h1, h2, h3 {font-weight: 600;}
.stButton > button {
    border-radius: 10px;
    padding: 0.4rem 1rem;
}
div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #eee;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)

DetectorFactory.seed = 0

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

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
    return str(text).replace('_x000D_', ' ').replace('\n', ' ').strip()

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, exclude):
    subset = [c for c in df.columns if c not in exclude]
    return df.drop_duplicates(subset=subset) if subset else df

def parse_keywords(text):
    return [k.strip() for k in text.split(",") if k.strip()]

def extract_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return "\n".join([s for s in sentences if any(k.lower() in s.lower() for k in keywords)])

def generate_cluster_summary(df):
    cluster_map = {}

    for c in df["Cluster"].unique():
        sample = df[df["Cluster"] == c]["Combined"].dropna().astype(str)

        if len(sample) == 0:
            cluster_map[c] = "Empty cluster"
        else:
            # take first 3 samples
            preview = sample.head(3).tolist()
            cluster_map[c] = " | ".join([p[:80] for p in preview])

    df["Cluster_Description"] = df["Cluster"].map(cluster_map)
    return df

# ==========================================
# STATUS BADGE
# ==========================================
def badge(status):
    colors = {
        "Not Run": "🟡",
        "Running": "🔵",
        "Done": "🟢",
        "Error": "🔴",
        "Skipped": "⚪"
    }
    return f"{colors.get(status, '🟡')} {status}"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Controls")

file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.info("Pipeline will process step-by-step")

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
    st.info("Upload a file to begin")
    st.stop()

# ==========================================
# DASHBOARD HEADER
# ==========================================
st.title("📊 Data Cleaner Pro")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Status", "Active")
col4.metric("Pipeline", "Ready")

st.markdown("---")

st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# STEP 1
# ==========================================
with st.expander("🧩 Step 1 — Combine Columns", expanded=True):

    cols = st.multiselect("Select columns", df.columns)

    c1, c2 = st.columns(2)
    run = c1.button("▶ Run Step 1")
    skip = c2.button("⏭ Skip Step 1")

    if run:
        df = create_combined(df, cols)
        st.session_state.data = df
        st.success(badge("Done"))

    if skip:
        st.warning(badge("Skipped"))

# ==========================================
# STEP 2
# ==========================================
with st.expander("🧹 Step 2 — Remove Duplicates"):

    exclude = st.multiselect("Exclude columns", df.columns)

    c1, c2 = st.columns(2)
    run = c1.button("▶ Run Step 2")
    skip = c2.button("⏭ Skip Step 2")

    if run:
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        st.success(badge("Done"))

    if skip:
        st.warning(badge("Skipped"))

# ==========================================
# STEP 3
# ==========================================
with st.expander("🔑 Step 3 — Keyword Matching"):

    num = st.number_input("Groups", 1, 10, 1)

    for i in range(num):

        st.markdown(f"### Group {i+1}")

        kw_text = st.text_input("Keywords (comma separated)", key=f"k{i}")
        tag_col = st.text_input("Tag column", f"Tags_{i+1}")

        # ⭐ NEW OPTION
        extract_sentence = st.checkbox(
            "Extract matching sentences?",
            key=f"sent_opt_{i}"
        )

        sent_col = st.text_input(
            "Sentence column name",
            f"Sent_{i+1}",
            disabled=not extract_sentence
        )

        if st.button(f"▶ Run Group {i+1}"):

            keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

            # TAGGING
            df[tag_col] = df["Combined"].apply(
                lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
            )

            # ⭐ OPTIONAL SENTENCE EXTRACTION
            if extract_sentence:

                def extract_sent(text):
                    sentences = re.split(r'(?<=[.!?])\s+', str(text))
                    return " ".join([
                        s for s in sentences
                        if any(k.lower() in s.lower() for k in keywords)
                    ])

                df[sent_col] = df["Combined"].apply(extract_sent)

            st.session_state.data = df
            st.success("✔ Done")

# ==========================================
# STEP 4
# ==========================================
with st.expander("🌍 Step 4 — Translation"):

    if st.button("▶ Run Translation"):
        translator = GoogleTranslator(source='auto', target='en')
        df["Translated"] = df["Combined"].apply(lambda x: translator.translate(str(x)[:2000]))
        st.session_state.data = df
        st.success(badge("Done"))

# ==========================================
# STEP 5
# ==========================================
with st.expander("💬 Step 5 — Sentiment"):

    source = st.radio("Source", ["Combined", "Translated"])

    brand_col = st.selectbox("Brand column", df.columns)

    if st.button("▶ Run Sentiment"):

        model = load_sentiment_model()

        results = []

        for _, row in df.iterrows():

            text = str(row[source])
            brand = str(row[brand_col])

            if brand.lower() not in text.lower():
                results.append("NO_MENTION")
                continue

            res = model(text[:512])[0]
            results.append(res["label"])

        df["Sentiment"] = results
        st.session_state.data = df
        st.success(badge("Done"))

# ==========================================
# STEP 6
# ==========================================
with st.expander("📦 Step 6 — Clustering"):

    threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

    if st.button("▶ Run Clustering"):

        model = load_model()

        emb = model.encode(
            df["Combined"].astype(str).tolist(),
            convert_to_numpy=True
        )
        emb = normalize(emb)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold
        )

        df["Cluster"] = clustering.fit_predict(emb)

        # ⭐ NEW: cluster description
        df = generate_cluster_summary(df)

        st.session_state.data = df
        st.success("✔ Clustering + Description Done")

# ==========================================
# OUTPUT
# ==========================================
st.markdown("---")
st.subheader("📦 Final Output")

st.dataframe(df, use_container_width=True)

st.download_button(
    "📥 Download Excel",
    data=BytesIO(),
    file_name="output.xlsx"
)
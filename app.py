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
            preview = sample.head(3).tolist()
            cluster_map[c] = " | ".join([p[:80] for p in preview])

    df["Cluster_Description"] = df["Cluster"].map(cluster_map)
    return df

# ==========================================
# PIPELINE
# ==========================================
def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    def tr(x):
        try:
            return translator.translate(str(x)[:2000])
        except:
            return x

    df["Translated"] = df["Combined"].apply(tr)
    return df

def cluster(df, threshold):
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

    return df

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

# ==========================================
# UI
# ==========================================
st.title("📊 Data Cleaner Pro")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

# LOAD DATA
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True

df = st.session_state.data

if df is None:
    st.info("Upload a file to start")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 1 (REQUIRED)
# ==========================================
st.header("🧩 Step 1 — Combine Columns (REQUIRED)")
st.warning("⚠️ Step 1 is required because all other steps depend on the Combined column.")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Step 1"):
    if len(cols) == 0:
        st.error("Please select at least one column")
    else:
        df = create_combined(df, cols)
        st.session_state.data = df
        st.success("Step 1 Completed ✔")

# STOP if no Combined
if "Combined" not in df.columns:
    st.stop()

# ==========================================
# STEP 2 (SKIPPABLE)
# ==========================================
st.header("🧹 Step 2 — Remove Duplicates")

skip2 = st.checkbox("Skip Step 2")

if not skip2:

    exclude = st.multiselect("Exclude columns", df.columns)

    if st.button("▶ Run Step 2"):
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        st.success("Step 2 Done ✔")

# ==========================================
# STEP 3 (SKIPPABLE)
# ==========================================
st.header("🔑 Step 3 — Keyword Matching")

skip3 = st.checkbox("Skip Step 3")

if not skip3:

    num = st.number_input("Keyword groups", 1, 10, 1)

    for i in range(num):

        st.subheader(f"Group {i+1}")

        kw_text = st.text_input("Keywords (comma separated)", key=f"k{i}")
        tag_col = st.text_input("Tag column", f"Tags_{i+1}")

        extract_sent = st.checkbox("Extract sentences?", key=f"s{i}")

        sent_col = st.text_input(
            "Sentence column",
            f"Sent_{i+1}",
            disabled=not extract_sent
        )

        if st.button(f"▶ Run Group {i+1}"):

            keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

            df[tag_col] = df["Combined"].apply(
                lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
            )

            if extract_sent:
                df[sent_col] = df["Combined"].apply(
                    lambda x: extract_sentences(x, keywords)
                )

            st.session_state.data = df
            st.success("Step 3 Done ✔")

# ==========================================
# STEP 4 (SKIPPABLE)
# ==========================================
st.header("🌍 Step 4 — Translation")

skip4 = st.checkbox("Skip Step 4")

if not skip4:

    if st.button("▶ Run Translation"):
        df = translate(df)
        st.session_state.data = df
        st.success("Step 4 Done ✔")

# ==========================================
# STEP 5 (SKIPPABLE)
# ==========================================
st.header("💬 Step 5 — Sentiment")

skip5 = st.checkbox("Skip Step 5")

if not skip5:

    source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])
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
        st.success("Step 5 Done ✔")

# ==========================================
# STEP 6 (SKIPPABLE)
# ==========================================
st.header("📦 Step 6 — Clustering")

skip6 = st.checkbox("Skip Step 6")

if not skip6:

    threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

    if st.button("▶ Run Clustering"):

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
        st.success("Step 6 Done ✔")

# ==========================================
# OUTPUT
# ==========================================
st.markdown("---")
st.subheader("📦 Final Output")

st.dataframe(df, use_container_width=True)

st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name="output.xlsx"
)

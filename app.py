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

# ==========================================
# CONFIG (Notion-style)
# ==========================================
st.set_page_config(
    page_title="Data Cleaner Pro",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
<style>
.block-container {padding: 2rem 2rem;}
h1, h2, h3 {font-weight: 600;}
.stButton > button {
    border-radius: 10px;
    padding: 0.4rem 1rem;
}
.stDataFrame {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

if "status" not in st.session_state:
    st.session_state.status = {}

# ==========================================
# STATUS SYSTEM
# ==========================================
def set_status(step, value):
    st.session_state.status[step] = value

def get_status(step):
    return st.session_state.status.get(step, "Not Run")

def icon(status):
    return {
        "Not Run": "🟡",
        "Running": "🔵",
        "Done": "🟢",
        "Error": "🔴",
        "Skipped": "⚪"
    }.get(status, "🟡")

def preview(df, title="Preview"):
    st.markdown(f"### 👀 {title}")
    st.dataframe(df.head(10), use_container_width=True)

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
def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
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
        cluster_map[c] = " | ".join(sample.head(3).tolist()) if len(sample) else "Empty"

    df["Cluster_Description"] = df["Cluster"].map(cluster_map)
    return df

def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    df["Translated"] = df["Combined"].apply(
        lambda x: translator.translate(str(x)[:2000])
    )
    return df

# ==========================================
# FIX: preserve hyperlinks safe export
# ==========================================
def to_excel(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)

    output.seek(0)
    return output

# ==========================================
# UI HEADER (Notion style)
# ==========================================
st.title("📊 Data Cleaner Pro")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

# ==========================================
# LOAD FILE
# ==========================================
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True

df = st.session_state.data

if df is None:
    st.info("Upload file to start")
    st.stop()

preview(df, "Raw Data")

# ==========================================
# STEP 1 (REQUIRED)
# ==========================================
st.header(f"🧩 Step 1 — Combine Columns {icon(get_status('step1'))}")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Step 1"):
    set_status("step1", "Running")
    try:
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("step1", "Done")
    except:
        set_status("step1", "Error")

preview(df, "After Step 1")

# ==========================================
# STEP 2
# ==========================================
st.header(f"🧹 Step 2 — Remove Duplicates {icon(get_status('step2'))}")

skip2 = st.checkbox("Skip Step 2")

if not skip2:
    exclude = st.multiselect("Exclude columns", df.columns)

    if st.button("▶ Run Step 2"):
        set_status("step2", "Running")
        try:
            df = remove_duplicates(df, exclude)
            st.session_state.data = df
            set_status("step2", "Done")
        except:
            set_status("step2", "Error")
else:
    set_status("step2", "Skipped")

preview(df, "After Step 2")

# ==========================================
# STEP 3
# ==========================================
st.header(f"🔑 Step 3 — Keyword Matching {icon(get_status('step3'))}")

skip3 = st.checkbox("Skip Step 3")

if not skip3:

    num = st.number_input("Groups", 1, 10, 1)

    for i in range(num):

        st.subheader(f"Group {i+1}")

        kw_text = st.text_input("Keywords (comma separated)", key=f"k{i}")
        tag_col = st.text_input("Tag column", f"Tags_{i+1}")

        extract_sent = st.checkbox("Extract sentences?", key=f"s{i}")
        sent_col = st.text_input("Sentence column", f"Sent_{i+1}")

        if st.button(f"▶ Run Group {i+1}"):

            set_status("step3", "Running")

            keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

            df[tag_col] = df["Combined"].apply(
                lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
            )

            if extract_sent:
                df[sent_col] = df["Combined"].apply(
                    lambda x: extract_sentences(x, keywords)
                )

            st.session_state.data = df
            set_status("step3", "Done")

else:
    set_status("step3", "Skipped")

preview(df, "After Step 3")

# ==========================================
# STEP 4
# ==========================================
st.header(f"🌍 Step 4 — Translation {icon(get_status('step4'))}")

skip4 = st.checkbox("Skip Step 4")

if not skip4:

    if st.button("▶ Run Step 4"):
        set_status("step4", "Running")
        try:
            df = translate(df)
            st.session_state.data = df
            set_status("step4", "Done")
        except:
            set_status("step4", "Error")

else:
    set_status("step4", "Skipped")

preview(df, "After Step 4")

# ==========================================
# STEP 5
# ==========================================
st.header(f"💬 Step 5 — Sentiment {icon(get_status('step5'))}")

skip5 = st.checkbox("Skip Step 5")

if not skip5:

    source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])
    brand_col = st.selectbox("Brand column", df.columns)

    if st.button("▶ Run Step 5"):

        set_status("step5", "Running")

        model = load_sentiment_model()

        results = []

        for _, row in df.iterrows():

            text = str(row[source])

            if str(row[brand_col]).lower() not in text.lower():
                results.append("NO_MENTION")
                continue

            results.append(model(text[:512])[0]["label"])

        df["Sentiment"] = results
        st.session_state.data = df

        set_status("step5", "Done")

else:
    set_status("step5", "Skipped")

preview(df, "After Step 5")

# ==========================================
# STEP 6
# ==========================================
st.header(f"📦 Step 6 — Clustering {icon(get_status('step6'))}")

skip6 = st.checkbox("Skip Step 6")

if not skip6:

    threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

    if st.button("▶ Run Step 6"):

        set_status("step6", "Running")

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

        set_status("step6", "Done")

else:
    set_status("step6", "Skipped")

preview(df, "After Step 6")

# ==========================================
# FINAL OUTPUT
# ==========================================
st.markdown("---")
st.subheader("📦 Final Output")

preview(df, "Final Data")

# ==========================================
# CUSTOM FILENAME EXPORT
# ==========================================
file_name = st.text_input("Output file name", "output.xlsx")

st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name=file_name
)
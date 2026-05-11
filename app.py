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

if "status" not in st.session_state:
    st.session_state.status = {
        "step1": "Not Run",
        "step2": "Not Run",
        "step3": "Not Run",
        "step4": "Not Run",
        "step5": "Not Run",
        "step6": "Not Run",
    }

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

# ==========================================
# MODELS
# ==========================================
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ==========================================
# HELPERS
# ==========================================
def clean_text(text):
    if pd.isnull(text):
        return text
    return str(text).replace("_x000D_", " ").replace("\n", " ").strip()

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def extract_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return " ".join([s for s in sentences if any(k.lower() in s.lower() for k in keywords)])

def generate_cluster_summary(df):
    summary = {}
    for c in df["Cluster"].unique():
        sample = df[df["Cluster"] == c]["Combined"].dropna().astype(str)
        summary[c] = " | ".join(sample.head(3).tolist())
    df["Cluster_Description"] = df["Cluster"].map(summary)
    return df

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font

def to_excel(df):
    wb = Workbook()
    ws = wb.active
    ws.title = "Output"

    # write dataframe rows
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):

            cell = ws.cell(row=r_idx, column=c_idx)

            # header row
            if r_idx == 1:
                cell.value = value
                continue

            # preserve hyperlinks
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                cell.value = value
                cell.hyperlink = value
                cell.font = Font(color="0000FF", underline="single")
            else:
                cell.value = value

    # auto column width (optional nice touch)
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                if cell.value:
                    max_len = max(max_len, len(str(cell.value)))
            except:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

# ==========================================
# UI HEADER
# ==========================================
st.title("📊 Data Cleaner Pro")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file and st.session_state.data is None:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    st.session_state.data = pd.read_excel(file, sheet_name=sheet)

df = st.session_state.data

if df is None:
    st.info("Upload file to start")
    st.stop()

st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 1
# ==========================================
st.header("🧩 Step 1 — Combine Columns (Required)")
cols = st.multiselect("Select columns", df.columns)

c1, c2 = st.columns(2)
run1 = c1.button("▶ Run Step 1", key="run1")
skip1 = c2.button("⏭ Skip Step 1", key="skip1")

if run1:
    if cols:
        set_status("step1", "Running")
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("step1", "Done")
    else:
        set_status("step1", "Error")

if skip1:
    set_status("step1", "Skipped")

if get_status("step1") == "Done":
    st.dataframe(df.head(), use_container_width=True)

if "Combined" not in df.columns:
    st.warning("Step 1 required")
    st.stop()

# ==========================================
# STEP 2
# ==========================================
st.header("🧹 Step 2 — Remove Duplicates")

c1, c2 = st.columns(2)
run2 = c1.button("▶ Run Step 2", key="run2")
skip2 = c2.button("⏭ Skip Step 2", key="skip2")

if run2:
    set_status("step2", "Running")
    df = remove_duplicates(df)
    st.session_state.data = df
    set_status("step2", "Done")

if skip2:
    set_status("step2", "Skipped")

if get_status("step2") == "Done":
    st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 3
# ==========================================
st.header("🔑 Step 3 — Keyword Matching")

num = st.number_input("Groups", 1, 10, 1)

for i in range(num):

    st.subheader(f"Group {i+1}")

    kw_file = st.file_uploader("Upload keyword file OR use manual", type=["xlsx"], key=f"file{i}")
    kw_text = st.text_input("Manual keywords", key=f"text{i}")

    tag_col = st.text_input("Tag column", f"Tags_{i+1}")

    extract_sent = st.checkbox("Extract sentences?", key=f"sent{i}")
    sent_col = st.text_input("Sentence column", f"Sent_{i+1}")

    c1, c2 = st.columns(2)
    run3 = c1.button("▶ Run", key=f"run3_{i}")
    skip3 = c2.button("⏭ Skip", key=f"skip3_{i}")

    if run3:

        set_status("step3", "Running")

        if kw_file and kw_text:
            st.error("Use only ONE input method")
            set_status("step3", "Error")
        else:
            keywords = []

            if kw_file:
                kw_df = pd.read_excel(kw_file)
                keywords = kw_df.iloc[:, 0].dropna().astype(str).tolist()
            else:
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

    if skip3:
        set_status("step3", "Skipped")

if get_status("step3") == "Done":
    st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 4
# ==========================================
st.header("🌍 Step 4 — Translation")

c1, c2 = st.columns(2)
run4 = c1.button("▶ Run Step 4", key="run4")
skip4 = c2.button("⏭ Skip Step 4", key="skip4")

if run4:
    set_status("step4", "Running")
    translator = GoogleTranslator(source="auto", target="en")
    df["Translated"] = df["Combined"].apply(lambda x: translator.translate(str(x)[:2000]))
    st.session_state.data = df
    set_status("step4", "Done")

if skip4:
    set_status("step4", "Skipped")

if get_status("step4") == "Done":
    st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 5
# ==========================================
st.header("💬 Step 5 — Sentiment")

source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])
brand_col = st.selectbox("Brand column", df.columns)

c1, c2 = st.columns(2)
run5 = c1.button("▶ Run Step 5", key="run5")
skip5 = c2.button("⏭ Skip Step 5", key="skip5")

if run5:
    set_status("step5", "Running")

    model = load_sentiment()
    results = []

    for _, row in df.iterrows():
        text = str(row[source])
        if str(row[brand_col]).lower() not in text.lower():
            results.append("NO_MENTION")
        else:
            results.append(model(text[:512])[0]["label"])

    df["Sentiment"] = results
    st.session_state.data = df
    set_status("step5", "Done")

if skip5:
    set_status("step5", "Skipped")

if get_status("step5") == "Done":
    st.dataframe(df.head(), use_container_width=True)

# ==========================================
# STEP 6
# ==========================================
st.header("📦 Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

c1, c2 = st.columns(2)
run6 = c1.button("▶ Run Step 6", key="run6")
skip6 = c2.button("⏭ Skip Step 6", key="skip6")

if run6:
    set_status("step6", "Running")

    model = load_model()
    emb = normalize(model.encode(df["Combined"].astype(str).tolist(), convert_to_numpy=True))

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

if skip6:
    set_status("step6", "Skipped")

if get_status("step6") == "Done":
    st.dataframe(df.head(), use_container_width=True)

# ==========================================
# OUTPUT
# ==========================================
st.markdown("---")

filename = st.text_input("Output filename", "output.xlsx")

st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name=filename
)
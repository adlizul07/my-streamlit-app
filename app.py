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
from openpyxl import load_workbook

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Data Cleaner Pro",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# LOAD EXCEL WITH HYPERLINKS (FIXED)
# ==========================================
def load_excel_with_hyperlinks(file, sheet_name):
    wb = load_workbook(file, data_only=False)
    ws = wb[sheet_name]

    headers = [cell.value for cell in ws[1]]

    rows = []

    for row in ws.iter_rows(min_row=2):
        row_data = []

        for cell in row:
            text = cell.value
            link = cell.hyperlink.target if cell.hyperlink else None

            row_data.append(text)

        rows.append(row_data)

    df = pd.DataFrame(rows, columns=headers)

    # store hyperlink separately (IMPORTANT FIX)
    if "Headline" in df.columns:
        df["Headline_Link"] = None

        col_idx = df.columns.get_loc("Headline")

        for i, row in enumerate(ws.iter_rows(min_row=2)):
            cell = row[col_idx]
            if cell.hyperlink:
                df.loc[i, "Headline_Link"] = cell.hyperlink.target

    return df


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


# ==========================================
# EXPORT (REMOVE LINK COLUMN BUT KEEP HYPERLINK)
# ==========================================
def to_excel(df):

    export_df = df.copy()

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False)

        workbook = writer.book
        sheet = writer.sheets["Sheet1"]

        # restore hyperlinks into Excel
        if "Headline_Link" in export_df.columns:

            col_idx = list(export_df.columns).index("Headline") + 1

            for row_idx, link in enumerate(export_df["Headline_Link"], start=2):
                if pd.notna(link):
                    cell = sheet.cell(row=row_idx, column=col_idx)
                    cell.hyperlink = link
                    cell.style = "Hyperlink"

        # remove helper column
        if "Headline_Link" in export_df.columns:
            export_df.drop(columns=["Headline_Link"], inplace=True)

    buffer.seek(0)
    return buffer


# ==========================================
# UI
# ==========================================
st.title("📊 Data Cleaner Pro")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file and st.session_state.data is None:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    st.session_state.data = load_excel_with_hyperlinks(file, sheet)

df = st.session_state.data

if df is None:
    st.info("Upload file to start")
    st.stop()

st.dataframe(df.head(), use_container_width=True)


# ==========================================
# STEP 1
# ==========================================
st.header("🧩 Step 1 — Combine Columns")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Step 1"):

    set_status("step1", "Running")

    df = create_combined(df, cols)

    st.session_state.data = df

    set_status("step1", "Done")


if "Combined" not in df.columns:
    st.stop()


# ==========================================
# STEP 2
# ==========================================
st.header("🧹 Step 2 — Remove Duplicates")

if st.button("▶ Run Step 2"):
    df = remove_duplicates(df)
    st.session_state.data = df
    set_status("step2", "Done")


# ==========================================
# STEP 3
# ==========================================
st.header("🔑 Step 3 — Keyword Matching")

kw_text = st.text_input("Keywords (comma separated)")

tag_col = st.text_input("Tag column", "Tags")

extract_sent = st.checkbox("Extract sentences?")
sent_col = st.text_input("Sentence column", "Sent")

if st.button("▶ Run Step 3"):

    keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

    df[tag_col] = df["Combined"].apply(
        lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
    )

    if extract_sent:
        df[sent_col] = df["Combined"].apply(
            lambda x: extract_sentences(x, keywords)
        )

    st.session_state.data = df


# ==========================================
# STEP 4
# ==========================================
st.header("🌍 Step 4 — Translation")

if st.button("▶ Run Step 4"):

    translator = GoogleTranslator(source="auto", target="en")

    df["Translated"] = df["Combined"].apply(
        lambda x: translator.translate(str(x)[:2000])
    )

    st.session_state.data = df


# ==========================================
# STEP 5
# ==========================================
st.header("💬 Step 5 — Sentiment")

source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])

brand_col = st.selectbox("Brand column", df.columns)

if st.button("▶ Run Step 5"):

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


# ==========================================
# STEP 6
# ==========================================
st.header("📦 Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

if st.button("▶ Run Step 6"):

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
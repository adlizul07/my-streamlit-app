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
# SESSION STATE (STATUS)
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

def remove_duplicates(df, exclude_cols):
    return df.drop_duplicates(subset=[c for c in df.columns if c not in exclude_cols])

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

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer


# ==========================================
# LOAD FILE (HYPERLINK SAFE)
# ==========================================
def load_excel(file, sheet):

    wb = load_workbook(file, data_only=False)
    ws = wb[sheet]

    headers = [cell.value for cell in ws[1]]

    rows = []

    for row in ws.iter_rows(min_row=2):

        row_data = []

        for cell in row:
            row_data.append(cell.value)

        rows.append(row_data)

    df = pd.DataFrame(rows, columns=headers)

    # ======================================
    # SAVE HEADLINE HYPERLINKS
    # ======================================
    if "Headline" in df.columns:

        df["Headline_Link"] = None

        headline_col_idx = list(df.columns).index("Headline")

        for i, row in enumerate(ws.iter_rows(min_row=2)):

            cell = row[headline_col_idx]

            if cell.hyperlink:
                df.loc[i, "Headline_Link"] = cell.hyperlink.target

    return df

# ==========================================
# UI
# ==========================================
st.title("📊 Data Cleaner Pro")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file and st.session_state.data is None:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    st.session_state.data = load_excel(file, sheet)

df = st.session_state.data

if df is None:
    st.info("Upload file to start")
    st.stop()

st.dataframe(df.head(), use_container_width=True)


# ==========================================
# STEP 1
# ==========================================
st.header(f"{icon(get_status('step1'))} Step 1 — Combine Columns")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Step 1"):

    if cols:
        set_status("step1", "Running")
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("step1", "Done")
    else:
        set_status("step1", "Error")

st.write(f"Status: {icon(get_status('step1'))} {get_status('step1')}")


# ==========================================
# STEP 2 (FIXED DROPDOWN EXCLUDE)
# ==========================================
st.header(f"{icon(get_status('step2'))} Step 2 — Remove Duplicates")

exclude_cols = st.multiselect(
    "Select columns to EXCLUDE from duplicate check",
    df.columns
)

if st.button("▶ Run Step 2"):

    set_status("step2", "Running")
    df = remove_duplicates(df, exclude_cols)
    st.session_state.data = df
    set_status("step2", "Done")

st.write(f"Status: {icon(get_status('step2'))} {get_status('step2')}")


# ==========================================
# STEP 3 (FIXED INPUT MODE)
# ==========================================
st.header(f"{icon(get_status('step3'))} Step 3 — Keyword Matching")

mode = st.radio(
    "Choose keyword input method",
    ["Upload File", "Manual Input"],
    horizontal=True
)

keywords = []

kw_file = None
kw_text = ""

if mode == "Upload File":
    kw_file = st.file_uploader("Upload keyword Excel file", type=["xlsx"])
else:
    kw_text = st.text_input("Enter keywords (comma separated)")


tag_col = st.text_input("Tag column", "Tags")
extract_sent = st.checkbox("Extract sentences?")
sent_col = st.text_input("Sentence column", "Sent")

if st.button("▶ Run Step 3"):

    set_status("step3", "Running")

    if mode == "Upload File" and kw_file:
        kw_df = pd.read_excel(kw_file)
        keywords = kw_df.iloc[:, 0].dropna().astype(str).tolist()

    elif mode == "Manual Input" and kw_text:
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

    else:
        st.error("Please provide keyword input")
        set_status("step3", "Error")
        st.stop()

    df[tag_col] = df["Combined"].apply(
        lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
    )

    if extract_sent:
        df[sent_col] = df["Combined"].apply(
            lambda x: extract_sentences(x, keywords)
        )

    st.session_state.data = df
    set_status("step3", "Done")

st.write(f"Status: {icon(get_status('step3'))} {get_status('step3')}")


# ==========================================
# STEP 4
# ==========================================
st.header(f"{icon(get_status('step4'))} Step 4 — Translation")

if st.button("▶ Run Step 4"):

    set_status("step4", "Running")

    translator = GoogleTranslator(source="auto", target="en")

    df["Translated"] = df["Combined"].apply(
        lambda x: translator.translate(str(x)[:2000])
    )

    st.session_state.data = df
    set_status("step4", "Done")

st.write(f"Status: {icon(get_status('step4'))} {get_status('step4')}")


# ==========================================
# STEP 5
# ==========================================
st.header(f"{icon(get_status('step5'))} Step 5 — Sentiment")

source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])
brand_col = st.selectbox("Brand column", df.columns)

if st.button("▶ Run Step 5"):

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

st.write(f"Status: {icon(get_status('step5'))} {get_status('step5')}")


# ==========================================
# STEP 6
# ==========================================
st.header(f"{icon(get_status('step6'))} Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

if st.button("▶ Run Step 6"):

    set_status("step6", "Running")

    model = load_model()

    emb = normalize(
        model.encode(df["Combined"].astype(str).tolist(), convert_to_numpy=True)
    )

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    df["Cluster"] = clustering.fit_predict(emb)

    # simple summary
    summary = {}
    for c in df["Cluster"].unique():
        summary[c] = " | ".join(df[df["Cluster"] == c]["Combined"].head(3).tolist())

    df["Cluster_Description"] = df["Cluster"].map(summary)

    st.session_state.data = df
    set_status("step6", "Done")

st.write(f"Status: {icon(get_status('step6'))} {get_status('step6')}")


# ==========================================
# OUTPUT (HYPERLINK PRESERVE FIX)
# ==========================================
st.markdown("---")

from openpyxl import Workbook

filename = st.text_input("Output filename", "output.xlsx")


def to_excel(df):

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")

        workbook = writer.book
        sheet = writer.sheets["Sheet1"]

# ======================================
# RESTORE HYPERLINKS (HEADLINE ONLY)
# ======================================
if "Headline" in df.columns:

    col_idx = list(df.columns).index("Headline") + 1

    for row_idx, row in enumerate(df.itertuples(index=False), start=2):

        cell = sheet.cell(row=row_idx, column=col_idx)

        if "Headline_Link" in df.columns:
            link = df.iloc[row_idx - 2]["Headline_Link"]
        else:
            link = getattr(row, "Headline", None)

        if (
            pd.notna(link)
            and isinstance(link, str)
            and link.startswith(("http://", "https://"))
        ):
            cell.hyperlink = link
            cell.style = "Hyperlink"

    buffer.seek(0)
    return buffer


st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name=filename
)
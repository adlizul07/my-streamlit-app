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
# POWER BI STYLE UI (CSS)
# ==========================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

.kpi {
    background: #1c1f26;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #2a2f3a;
    color: white;
}

.step-card {
    background: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 12px;
    border: 1px solid #2a2f3a;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

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
# PREVIEW
# ==========================================
def show_preview(step, df):
    if get_status(step) == "Done":
        st.dataframe(df.head(), use_container_width=True)

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

# ==========================================
# LOAD EXCEL
# ==========================================
def load_excel(file, sheet):
    wb = load_workbook(file, data_only=False)
    ws = wb[sheet]

    headers = [cell.value for cell in ws[1]]

    rows = []
    for row in ws.iter_rows(min_row=2):
        rows.append([cell.value for cell in row])

    df = pd.DataFrame(rows, columns=headers)

    if "Headline" in df.columns:
        df["Headline_Link"] = None
        idx = list(df.columns).index("Headline")

        for i, row in enumerate(ws.iter_rows(min_row=2)):
            cell = row[idx]
            if cell.hyperlink:
                df.loc[i, "Headline_Link"] = cell.hyperlink.target

    return df

# ==========================================
# UI HEADER (POWER BI STYLE)
# ==========================================
st.title("📊 Data Cleaner Pro Dashboard")

# KPI CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='kpi'>Step 1<br>{get_status('step1')}</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='kpi'>Step 2<br>{get_status('step2')}</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='kpi'>Step 3<br>{get_status('step3')}</div>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# LOAD FILE
# ==========================================
file = st.file_uploader("Upload Excel File", type=["xlsx"])

df = None

if file and st.session_state.data is None:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    df = load_excel(file, sheet)

    st.session_state.data = df

df = st.session_state.data

if df is None:
    st.info("Upload file to start")
    st.stop()

st.markdown("### 📊 Data Preview")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# ==========================================
# STEP 1
# ==========================================
st.markdown("## 🧩 Step 1 — Combine Columns")

with st.container():
    cols = st.multiselect("Select columns", df.columns)

    if st.button("▶ Run Step 1"):
        if cols:
            set_status("step1", "Running")
            df = create_combined(df, cols)
            st.session_state.data = df
            set_status("step1", "Done")
        else:
            set_status("step1", "Error")

    st.info(f"{icon(get_status('step1'))} {get_status('step1')}")
    show_preview("step1", df)

st.markdown("---")

# ==========================================
# STEP 2
# ==========================================
st.markdown("## 🧹 Step 2 — Remove Duplicates")

exclude_cols = st.multiselect("Exclude columns", df.columns)

c1, c2 = st.columns(2)

with c1:
    if st.button("▶ Run Step 2"):
        set_status("step2", "Running")
        df = remove_duplicates(df, exclude_cols)
        st.session_state.data = df
        set_status("step2", "Done")

with c2:
    if st.button("⏭ Skip Step 2"):
        set_status("step2", "Skipped")

st.info(f"{icon(get_status('step2'))} {get_status('step2')}")
show_preview("step2", df)

st.markdown("---")

# ==========================================
# STEP 3
# ==========================================
st.markdown("## 🔎 Step 3 — Keyword Matching")

mode = st.radio("Input Mode", ["Upload File", "Manual Input"], horizontal=True)

keywords = []
display_map = {}

kw_file = None
kw_text = ""

keyword_col = None
display_col = None

if mode == "Upload File":
    kw_file = st.file_uploader("Keyword File", type=["xlsx"])

    if kw_file:
        preview_kw_df = pd.read_excel(kw_file)
        st.dataframe(preview_kw_df.head())

        if len(preview_kw_df.columns) > 1:
            keyword_col = st.selectbox("Keyword column", preview_kw_df.columns)
            display_col = st.selectbox("Display column", preview_kw_df.columns)
        else:
            keyword_col = display_col = preview_kw_df.columns[0]

else:
    kw_text = st.text_input("Enter keywords (comma separated)")

tag_col = st.text_input("Tag column", "Tags")
sent_col = st.text_input("Sentence column", "Sent")

def exact_match(text, keyword):
    pattern = r'(?i)(?<!\w)' + re.escape(keyword) + r'(?!\w)'
    return re.search(pattern, str(text)) is not None

c1, c2 = st.columns(2)

with c1:
    if st.button("▶ Run Step 3"):
        set_status("step3", "Running")

        if mode == "Upload File" and kw_file:
            kw_df = pd.read_excel(kw_file)
            keywords = kw_df[keyword_col].astype(str).tolist()
            display_map = dict(zip(keywords, kw_df[display_col].astype(str)))

        else:
            keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
            display_map = {k: k for k in keywords}

        def match(text):
            return ", ".join([display_map[k] for k in keywords if exact_match(text, k)])

        df[tag_col] = df["Combined"].apply(match)

        st.session_state.data = df
        set_status("step3", "Done")

with c2:
    if st.button("⏭ Skip Step 3"):
        set_status("step3", "Skipped")

st.info(f"{icon(get_status('step3'))} {get_status('step3')}")
show_preview("step3", df)

st.markdown("---")

# ==========================================
# STEP 4
# ==========================================
st.markdown("## 🌐 Step 4 — Translation")

c1, c2 = st.columns(2)

if c1.button("▶ Run Step 4"):
    set_status("step4", "Running")

    translator = GoogleTranslator(source="auto", target="en")
    df["Translated"] = df["Combined"].apply(lambda x: translator.translate(str(x)[:2000]))

    st.session_state.data = df
    set_status("step4", "Done")

if c2.button("⏭ Skip Step 4"):
    set_status("step4", "Skipped")

st.info(f"{icon(get_status('step4'))} {get_status('step4')}")
show_preview("step4", df)

st.markdown("---")

# ==========================================
# STEP 5
# ==========================================
st.markdown("## ❤️ Step 5 — Sentiment")

source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])

brand_col = st.selectbox("Brand column", df.columns)

c1, c2 = st.columns(2)

if c1.button("▶ Run Step 5"):
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

if c2.button("⏭ Skip Step 5"):
    set_status("step5", "Skipped")

st.info(f"{icon(get_status('step5'))} {get_status('step5')}")
show_preview("step5", df)

st.markdown("---")

# ==========================================
# STEP 6
# ==========================================
st.markdown("## 🧠 Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

c1, c2 = st.columns(2)

if c1.button("▶ Run Step 6"):
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

    st.session_state.data = df
    set_status("step6", "Done")

if c2.button("⏭ Skip Step 6"):
    set_status("step6", "Skipped")

st.info(f"{icon(get_status('step6'))} {get_status('step6')}")
show_preview("step6", df)

st.markdown("---")

# ==========================================
# OUTPUT
# ==========================================
st.markdown("## 📤 Export")

filename = st.text_input("Filename", "output.xlsx")

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name=filename
)
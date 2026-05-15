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
    page_title="Insights Bibik Pro",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# CUSTOM CSS — Dark Industrial Dashboard
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- ROOT TOKENS ---- */
:root {
    --bg-base:      #0d0f12;
    --bg-card:      #13161b;
    --bg-raised:    #1a1e26;
    --border:       #252932;
    --border-glow:  #2e7dff44;
    --accent:       #2e7dff;
    --accent-soft:  #2e7dff18;
    --accent-green: #00d48f;
    --accent-amber: #f5a623;
    --accent-red:   #ff4757;
    --accent-grey:  #6b7280;
    --text-primary: #e8eaf0;
    --text-secondary: #8b92a5;
    --text-muted:   #4a5162;
    --radius:       10px;
    --radius-lg:    16px;
}

/* ---- GLOBAL RESET ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* ---- HIDE STREAMLIT CHROME ---- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---- MAIN CONTENT PADDING ---- */
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1280px;
}

/* ---- APP HEADER ---- */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 36px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.app-header-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 24px #2e7dff55;
}
.app-header-title {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}
.app-header-sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 2px;
}
.app-header-badge {
    margin-left: auto;
    background: var(--accent-soft);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 100px;
    letter-spacing: 0.5px;
}

/* ---- STEP CARD ---- */
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    transition: border-color 0.2s;
}
.step-card:hover { border-color: #2e3440; }
.step-card.active { border-color: var(--accent); box-shadow: 0 0 0 1px var(--border-glow), 0 4px 32px #2e7dff12; }
.step-card.done  { border-color: #00d48f33; }
.step-card.error { border-color: #ff475733; }

.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}
.step-number {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    background: var(--bg-raised);
    border: 1px solid var(--border);
    padding: 3px 9px;
    border-radius: 6px;
    letter-spacing: 1px;
}
.step-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
}
.step-divider {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
}

/* ---- STATUS PILL ---- */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 100px;
    border: 1px solid;
    margin-top: 14px;
}
.status-not-run { color: var(--accent-amber); border-color: #f5a62355; background: #f5a62310; }
.status-running  { color: var(--accent); border-color: #2e7dff55; background: var(--accent-soft); }
.status-done     { color: var(--accent-green); border-color: #00d48f55; background: #00d48f10; }
.status-error    { color: var(--accent-red); border-color: #ff475755; background: #ff475710; }
.status-skipped  { color: var(--accent-grey); border-color: #6b728055; background: #6b728010; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: currentColor; }

/* ---- UPLOAD ZONE ---- */
[data-testid="stFileUploader"] {
    background: var(--bg-raised) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ---- BUTTONS ---- */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    transition: all 0.15s !important;
    border: 1px solid !important;
}

/* Primary (Run) buttons */
.stButton > button[kind="primary"],
.stButton > button:not([kind="secondary"]) {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
    box-shadow: 0 2px 12px #2e7dff30 !important;
}
.stButton > button:not([kind="secondary"]):hover {
    background: #4a92ff !important;
    box-shadow: 0 4px 20px #2e7dff55 !important;
    transform: translateY(-1px);
}

/* Skip buttons */
.stButton > button[kind="secondary"] {
    background: var(--bg-raised) !important;
    border-color: var(--border) !important;
    color: var(--text-secondary) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-grey) !important;
    color: var(--text-primary) !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    width: 100%;
    background: var(--accent-green) !important;
    border-color: var(--accent-green) !important;
    color: #000 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 20px #00d48f30 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #00e89a !important;
    box-shadow: 0 6px 28px #00d48f55 !important;
    transform: translateY(-1px);
}

/* ---- INPUTS ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--border-glow) !important;
}

/* ---- RADIO ---- */
.stRadio > label { color: var(--text-secondary) !important; font-size: 13px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text-primary) !important; }

/* ---- SLIDER ---- */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background: var(--accent) !important;
}

/* ---- CHECKBOX ---- */
.stCheckbox [data-baseweb="checkbox"] { gap: 8px !important; }

/* ---- DATAFRAME ---- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] table { background: var(--bg-raised) !important; }
[data-testid="stDataFrame"] th { background: var(--bg-card) !important; color: var(--text-secondary) !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }
[data-testid="stDataFrame"] td { color: var(--text-primary) !important; font-size: 13px !important; }

/* ---- LABELS ---- */
label, .stSelectbox label, .stMultiSelect label,
.stTextInput label, .stTextArea label, .stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}

/* ---- JSON ---- */
.stJson { background: var(--bg-raised) !important; border-radius: var(--radius) !important; border: 1px solid var(--border) !important; }

/* ---- ALERTS ---- */
.stAlert { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; }
.stInfo { background: var(--accent-soft) !important; border-color: #2e7dff33 !important; }
.stWarning { background: #f5a62312 !important; border-color: #f5a62333 !important; }

/* ---- DIVIDER ---- */
hr { border-color: var(--border) !important; }

/* ---- SECTION LABEL ---- */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 16px;
}

/* ---- OUTPUT CARD ---- */
.output-card {
    background: linear-gradient(135deg, #0f1a2e 0%, #0d1520 100%);
    border: 1px solid #2e7dff44;
    border-radius: var(--radius-lg);
    padding: 32px;
    text-align: center;
    margin-top: 8px;
}
.output-card h3 {
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.output-card p {
    color: var(--text-muted);
    font-size: 13px;
    margin-bottom: 24px;
}

/* ---- PROGRESS BAR ---- */
.pipeline-progress {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 32px;
}
.pipeline-step {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: var(--border);
    transition: background 0.3s;
}
.pipeline-step.done    { background: var(--accent-green); }
.pipeline-step.active  { background: var(--accent); }
.pipeline-step.skipped { background: var(--accent-grey); }
.pipeline-step.error   { background: var(--accent-red); }

</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "status" not in st.session_state:
    st.session_state.status = {
        "step1": "Not Run", "step2": "Not Run", "step3": "Not Run",
        "step4": "Not Run", "step5": "Not Run", "step6": "Not Run",
    }

def set_status(step, value):
    st.session_state.status[step] = value

def get_status(step):
    return st.session_state.status.get(step, "Not Run")

def status_pill(step):
    s = get_status(step)
    cls_map = {
        "Not Run": "status-not-run", "Running": "status-running",
        "Done": "status-done", "Error": "status-error", "Skipped": "status-skipped"
    }
    dot_map = {
        "Not Run": "●", "Running": "◉", "Done": "●", "Error": "✕", "Skipped": "—"
    }
    cls = cls_map.get(s, "status-not-run")
    dot = dot_map.get(s, "●")
    return f'<span class="status-pill {cls}"><span class="status-dot">{dot}</span>{s}</span>'

def step_card_class(step):
    s = get_status(step)
    return {"Running": "active", "Done": "done", "Error": "error"}.get(s, "")

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

def generate_cluster_summary(df):
    summary = {}
    for c in df["Cluster"].unique():
        sample = df[df["Cluster"] == c]["Combined"].dropna().astype(str)
        summary[c] = " | ".join(sample.head(3).tolist())
    df["Cluster_Description"] = df["Cluster"].map(summary)
    return df

def exact_keyword_match(text, keyword):
    text_tokens = re.findall(r'\b\w+\b', str(text).lower())
    return keyword.lower() in text_tokens

def load_excel(file, sheet):
    wb = load_workbook(file, data_only=False)
    ws = wb[sheet]

    headers = [cell.value for cell in ws[1]]

    # ==========================================
    # HEADER VALIDATION
    # ==========================================
    cleaned_headers = []

    for h in headers:
        if h is None or str(h).strip() == "":
            st.error("❌ Wrong header detected. Please check your data.")
            st.stop()

        cleaned_headers.append(str(h).strip())

    # Check duplicate headers
    if len(cleaned_headers) != len(set(cleaned_headers)):
        st.error("❌ Wrong header detected. Please check your data.")
        st.stop()

    # ==========================================
    # LOAD DATA
    # ==========================================
    rows = []

    for row in ws.iter_rows(min_row=2):
        rows.append([cell.value for cell in row])

    df = pd.DataFrame(rows, columns=cleaned_headers)

    # ==========================================
    # PRESERVE HYPERLINKS
    # ==========================================
    if "Headline" in df.columns:
        df["Headline_Link"] = None
        headline_col_idx = list(df.columns).index("Headline")

        for i, row in enumerate(ws.iter_rows(min_row=2)):
            cell = row[headline_col_idx]

            if cell.hyperlink:
                df.loc[i, "Headline_Link"] = cell.hyperlink.target

    return df
def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
        workbook = writer.book
        sheet = writer.sheets["Sheet1"]
        if "Headline" in df.columns and "Headline_Link" in df.columns:
            col_idx = list(df.columns).index("Headline") + 1
            for row_idx in range(len(df)):
                link = df.iloc[row_idx]["Headline_Link"]
                if pd.notna(link) and isinstance(link, str) and link.startswith(("http://", "https://")):
                    cell = sheet.cell(row=row_idx + 2, column=col_idx)
                    cell.hyperlink = link
                    cell.style = "Hyperlink"
        if "Headline_Link" in df.columns:
            helper_col = list(df.columns).index("Headline_Link") + 1
            sheet.delete_cols(helper_col)
    buffer.seek(0)
    return buffer

# ==========================================
# APP HEADER
# ==========================================
st.markdown("""
<div class="app-header">
    <div class="app-header-icon">📊</div>
    <div>
        <div class="app-header-title">INSIGHTS BIBIK PRO</div>
        <div class="app-header-sub">Media intelligence pipeline · NLP + Clustering + Sentiment</div>
    </div>
    <div class="app-header-badge">v2.0</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PIPELINE PROGRESS BAR
# ==========================================
steps = ["step1", "step2", "step3", "step4", "step5", "step6"]
step_divs = ""
for s in steps:
    status = get_status(s)
    cls = {"Done": "done", "Running": "active", "Skipped": "skipped", "Error": "error"}.get(status, "")
    step_divs += f'<div class="pipeline-step {cls}"></div>'

st.markdown(f"""
<div class="pipeline-progress">{step_divs}</div>
<p style="font-family:'Space Mono',monospace;font-size:11px;color:#4a5162;margin-bottom:28px;letter-spacing:1px;">
  PIPELINE · 6 STEPS
</p>
""", unsafe_allow_html=True)

# ==========================================
# FILE UPLOAD
# ==========================================
st.markdown('<div class="section-label">Data Source</div>', unsafe_allow_html=True)

file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx"])

if file and st.session_state.data is None:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    st.session_state.data = load_excel(file, sheet)

df = st.session_state.data

if df is None:
    st.markdown("""
    <div style="text-align:center;padding:48px 24px;color:#4a5162;font-size:14px;">
        ⬆ &nbsp; Upload an Excel file to begin your pipeline
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown('<div class="section-label" style="margin-top:24px;">Preview — Raw Data</div>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# STEP 1 — COMBINE COLUMNS
# ==========================================
card_cls = step_card_class("step1")
st.markdown(f'<div class="step-card {card_cls}">', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-number">STEP 01</span><span class="step-title">Combine Columns</span></div>', unsafe_allow_html=True)

cols = st.multiselect("Select columns to combine into a single `Combined` field", df.columns)

if st.button("▶ Run Step 1", key="run1"):
    if cols:
        set_status("step1", "Running")
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("step1", "Done")
    else:
        set_status("step1", "Error")

st.markdown(status_pill("step1"), unsafe_allow_html=True)
show_preview("step1", df)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 2 — REMOVE DUPLICATES
# ==========================================
card_cls = step_card_class("step2")
st.markdown(f'<div class="step-card {card_cls}">', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-number">STEP 02</span><span class="step-title">Remove Duplicates</span></div>', unsafe_allow_html=True)

exclude_cols = st.multiselect("Exclude columns from duplicate check", df.columns)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run Step 2", key="run2"):
        set_status("step2", "Running")
        df = remove_duplicates(df, exclude_cols)
        st.session_state.data = df
        set_status("step2", "Done")
with col2:
    if st.button("⏭ Skip", key="skip2"):
        set_status("step2", "Skipped")

st.markdown(status_pill("step2"), unsafe_allow_html=True)
show_preview("step2", df)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 3 — KEYWORD MATCHING
# ==========================================
card_cls = step_card_class("step3")
st.markdown(f'<div class="step-card {card_cls}">', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-number">STEP 03</span><span class="step-title">Keyword Matching — Multi-Group</span></div>', unsafe_allow_html=True)

if "keyword_groups" not in st.session_state:
    st.session_state.keyword_groups = []

mode = st.radio("Input method", ["Upload File", "Manual Input"], horizontal=True)

st.markdown('<div class="step-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Add Keyword Group</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    group_name = st.text_input("Group Name", "Group1")
    tag_col = st.text_input("Output Column Name", "Tags")
with col_b:
    extract_sent = st.checkbox("Extract matched sentences")
    sent_col = st.text_input("Sentence column name", "Sent")

keywords = []
display_map = {}
kw_file = None
kw_text = ""
keyword_col = None
display_col = None

if mode == "Upload File":
    kw_file = st.file_uploader("Upload keyword Excel file", type=["xlsx"], key="kw_upload")
    if kw_file:
        preview_kw_df = pd.read_excel(kw_file)
        st.dataframe(preview_kw_df.head(), use_container_width=True)
        if len(preview_kw_df.columns) > 1:
            kc1, kc2 = st.columns(2)
            with kc1: keyword_col = st.selectbox("Keyword column", preview_kw_df.columns)
            with kc2: display_col = st.selectbox("Display column", preview_kw_df.columns)
        else:
            keyword_col = preview_kw_df.columns[0]
            display_col = preview_kw_df.columns[0]
else:
    kw_text = st.text_area("Enter keywords (comma separated)", placeholder="e.g. inflation, interest rate, GDP growth")

if st.button("➕ Add Keyword Group", key="add_group"):
    if mode == "Upload File" and kw_file:
        kw_df = pd.read_excel(kw_file).dropna(subset=[keyword_col])
        keywords = kw_df[keyword_col].astype(str).tolist()
        display_values = kw_df[display_col].astype(str).tolist()
        display_map = dict(zip(keywords, display_values))
    elif mode == "Manual Input" and kw_text:
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
        display_map = {k: k for k in keywords}
    else:
        st.error("Please provide keywords before adding a group.")
        st.stop()

    st.session_state.keyword_groups.append({
        "group": group_name, "keywords": keywords,
        "map": display_map, "output_col": tag_col
    })
    st.success(f"✓ Group **{group_name}** added — {len(keywords)} keywords")

if st.session_state.keyword_groups:
    st.markdown('<div class="section-label" style="margin-top:16px;">Active Groups</div>', unsafe_allow_html=True)
    st.json(st.session_state.keyword_groups)

st.markdown('<div class="step-divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run All Keyword Groups", key="run3"):
        set_status("step3", "Running")
        for g in st.session_state.keyword_groups:
            def get_matches(text, grp=g):
                matched = [grp["map"][k] for k in grp["keywords"] if exact_keyword_match(text, k)]
                return ", ".join(matched)
            df[g["output_col"]] = df["Combined"].apply(get_matches)

        if extract_sent:
            def extract_matching_sentences(text):
                sentences = re.split(r'(?<=[.!?])\s+', str(text))
                matched_sentences = []
                for s in sentences:
                    for g in st.session_state.keyword_groups:
                        for k in g["keywords"]:
                            if exact_keyword_match(s, k):
                                matched_sentences.append(s)
                                break
                return " ".join(matched_sentences)
            df[sent_col] = df["Combined"].apply(extract_matching_sentences)

        st.session_state.data = df
        set_status("step3", "Done")
with col2:
    if st.button("⏭ Skip", key="skip3"):
        set_status("step3", "Skipped")

st.markdown(status_pill("step3"), unsafe_allow_html=True)
show_preview("step3", df)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 4 SAFETY CHECK
# ==========================================
if "Combined" not in df.columns:
    st.warning("⚠ Complete Step 1 before proceeding.")
    st.stop()

# ==========================================
# STEP 4 — TRANSLATION
# ==========================================
card_cls = step_card_class("step4")
st.markdown(f'<div class="step-card {card_cls}">', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-number">STEP 04</span><span class="step-title">Auto-Translation → English</span></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8b92a5;font-size:13px;margin-bottom:16px;">Translates the Combined field to English using Google Translate (auto-detect source language).</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run Step 4", key="run4"):
        set_status("step4", "Running")
        translator = GoogleTranslator(source="auto", target="en")
        df["Translated"] = df["Combined"].apply(lambda x: translator.translate(str(x)[:2000]))
        st.session_state.data = df
        set_status("step4", "Done")
with col2:
    if st.button("⏭ Skip", key="skip4"):
        set_status("step4", "Skipped")

st.markdown(status_pill("step4"), unsafe_allow_html=True)
show_preview("step4", df)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 5 — SENTIMENT
# ==========================================
card_cls = step_card_class("step5")
st.markdown(f'<div class="step-card {card_cls}">', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-number">STEP 05</span><span class="step-title">Sentiment Analysis</span></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8b92a5;font-size:13px;margin-bottom:16px;">Powered by FinBERT — optimised for financial & business text.</p>', unsafe_allow_html=True)

source_options = ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"]
col_a, col_b = st.columns(2)
with col_a:
    source = st.radio("Analyse from", source_options, horizontal=True)
with col_b:
    brand_col = st.selectbox("Brand / entity column", df.columns)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run Step 5", key="run5"):
        set_status("step5", "Running")
        model = load_sentiment()
        results = []
        for _, row in df.iterrows():
            text = str(row[source]) if pd.notna(row[source]) else ""
            if str(row[brand_col]).lower() not in text.lower():
                results.append("NO_MENTION")
            else:
                results.append(model(text[:512])[0]["label"])
        df["Sentiment"] = results
        st.session_state.data = df
        set_status("step5", "Done")
with col2:
    if st.button("⏭ Skip", key="skip5"):
        set_status("step5", "Skipped")

st.markdown(status_pill("step5"), unsafe_allow_html=True)
show_preview("step5", df)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# STEP 6 — CLUSTERING
# ==========================================
card_cls = step_card_class("step6")
st.markdown(f'<div class="step-card {card_cls}">', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-number">STEP 06</span><span class="step-title">Semantic Clustering</span></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#8b92a5;font-size:13px;margin-bottom:16px;">Agglomerative clustering using multilingual sentence embeddings (cosine similarity).</p>', unsafe_allow_html=True)

threshold = st.slider("Distance threshold (lower = stricter clusters)", 0.25, 0.35, 0.28, step=0.01)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Run Step 6", key="run6"):
        set_status("step6", "Running")
        model = load_model()
        emb = normalize(model.encode(df["Combined"].astype(str).tolist(), convert_to_numpy=True))
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
        )
        df["Cluster"] = clustering.fit_predict(emb)
        summary = {}
        for c in df["Cluster"].unique():
            summary[c] = " | ".join(df[df["Cluster"] == c]["Combined"].head(3).tolist())
        df["Cluster_Description"] = df["Cluster"].map(summary)
        st.session_state.data = df
        set_status("step6", "Done")
with col2:
    if st.button("⏭ Skip", key="skip6"):
        set_status("step6", "Skipped")

st.markdown(status_pill("step6"), unsafe_allow_html=True)
show_preview("step6", df)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# OUTPUT
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown('<div class="output-card">', unsafe_allow_html=True)
st.markdown('<h3>📥 Export Results</h3>', unsafe_allow_html=True)
st.markdown('<p>Download your processed dataset as a formatted Excel file</p>', unsafe_allow_html=True)

filename = st.text_input("Filename", "output.xlsx", label_visibility="collapsed")

st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name=filename
)
st.markdown('</div>', unsafe_allow_html=True)
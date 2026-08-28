import streamlit as st
import pandas as pd
import numpy as np
import io, requests
from io import BytesIO
from bs4 import BeautifulSoup
import pdfplumber
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from openpyxl import load_workbook

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Insights Copilot",
    layout="wide",
    page_icon="\U0001F4CA",
    initial_sidebar_state="expanded",
)

STEPS = [
    ("step1", "Extract text from links"),
    ("step2", "Combine columns"),
    ("step3", "Remove duplicates"),
    ("step4", "Translate to English"),
    ("step5", "Semantic clustering"),
]

# ==========================================
# STYLES — light, one typeface, flat surfaces
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg-base:        #0d0f12;
    --bg-raised:      #13161b;
    --border:         #252932;
    --border-strong:  #333846;
    --accent:         #2e7dff;
    --accent-soft:    #2e7dff18;
    --ok:             #00d48f;
    --warn:           #f5a623;
    --err:            #ff4757;
    --muted:          #4a5162;
    --text:           #e8eaf0;
    --text-2:         #8b92a5;
    --radius:         8px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-raised) !important;
    border-right: 1px solid var(--border);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.block-container { padding: 2rem 3rem 4rem !important; max-width: 1180px; }

/* ---- Header ---- */
.app-header { padding: 8px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
.app-header-title { font-size: 22px; font-weight: 700; letter-spacing: -0.4px; }
.app-header-sub { font-size: 13px; color: var(--text-2); margin-top: 2px; }

/* ---- Pipeline strip ---- */
.pipeline-progress { display: flex; gap: 4px; margin: 4px 0 6px; }
.pipeline-step { flex: 1; height: 4px; border-radius: 2px; background: var(--border); }
.pipeline-step.done   { background: var(--ok); }
.pipeline-step.active { background: var(--accent); }
.pipeline-step.error  { background: var(--err); }
.pipeline-caption { font-size: 12px; color: var(--muted); margin-bottom: 22px; }

/* ---- Status pill ---- */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; padding: 3px 10px; border-radius: 100px;
    border: 1px solid; margin-top: 10px;
}
.status-not-run { color: var(--warn); border-color: #f5a62355; background: #f5a62310; }
.status-running { color: var(--accent); border-color: #2e7dff55; background: var(--accent-soft); }
.status-done    { color: var(--ok); border-color: #00d48f55; background: #00d48f10; }
.status-error   { color: var(--err); border-color: #ff475755; background: #ff475710; }

/* ---- Section label ---- */
.section-label { font-size: 12px; font-weight: 600; color: var(--text-2); margin-bottom: 10px; }
.step-note { color: var(--text-2); font-size: 13px; margin-bottom: 14px; }

/* ---- Expanders (one per step) ---- */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--bg-base) !important;
    margin-bottom: 8px;
}
[data-testid="stExpander"] summary { font-weight: 500 !important; font-size: 14px !important; }

/* ---- Buttons ---- */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: var(--radius) !important;
    padding: 7px 18px !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--bg-base) !important;
    color: var(--text) !important;
    box-shadow: none !important;
    transition: background 0.12s, border-color 0.12s;
}
.stButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover { background: #4a92ff !important; border-color: #4a92ff !important; color: #fff !important; }

[data-testid="stDownloadButton"] > button {
    width: 100%;
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #fff !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: var(--radius) !important;
    box-shadow: none !important;
}

/* ---- Inputs ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-base) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; }

/* Value text inside selects — without this it inherits the base theme colour
   and can render white on white. */
.stSelectbox div[data-baseweb="select"] *,
.stMultiSelect div[data-baseweb="select"] * {
    color: var(--text) !important;
}
.stSelectbox div[data-baseweb="select"] svg,
.stMultiSelect div[data-baseweb="select"] svg { fill: var(--text-2) !important; }

/* Selected chips in a multiselect */
.stMultiSelect [data-baseweb="tag"] {
    background: var(--accent-soft) !important;
    border: 1px solid #2e7dff55 !important;
}
.stMultiSelect [data-baseweb="tag"] span { color: var(--accent) !important; }

/* The dropdown menu itself renders in a portal outside the widget */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background: var(--bg-base) !important;
    color: var(--text) !important;
}
div[data-baseweb="popover"] li:hover { background: var(--accent-soft) !important; }

label, .stSelectbox label, .stMultiSelect label,
.stTextInput label, .stCheckbox label, .stSlider label {
    color: var(--text-2) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; overflow: hidden !important; }
[data-testid="stDataFrame"] th { background: var(--bg-raised) !important; color: var(--text-2) !important; font-size: 12px !important; }
[data-testid="stDataFrame"] td { color: var(--text) !important; font-size: 13px !important; }

/* ---- Metrics row ---- */
[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: var(--text-2) !important; }

/* ---- Progress ---- */
[data-testid="stProgress"] > div > div > div > div { background-color: var(--accent) !important; }

hr { border-color: var(--border) !important; }
.stAlert { border-radius: var(--radius) !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None
if "status" not in st.session_state:
    st.session_state.status = {k: "Not run" for k, _ in STEPS}
if "undo" not in st.session_state:
    st.session_state.undo = None          # (label, dataframe copy, status copy)
if "export_buffer" not in st.session_state:
    st.session_state.export_buffer = None


def set_status(step, value):
    st.session_state.status[step] = value


def get_status(step):
    return st.session_state.status.get(step, "Not run")


def status_pill(step):
    s = get_status(step)
    cls = {"Not run": "status-not-run", "Running": "status-running",
           "Done": "status-done", "Error": "status-error"}.get(s, "status-not-run")
    dot = {"Not run": "\u25CB", "Running": "\u25C9", "Done": "\u25CF", "Error": "\u2715"}.get(s, "\u25CB")
    return f'<span class="status-pill {cls}">{dot} {s}</span>'


def save_undo(label):
    st.session_state.undo = (
        label,
        st.session_state.data.copy(deep=True),
        dict(st.session_state.status),
    )


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
    return str(text).replace("_x000D_", " ").replace("\n", " ").strip()


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _html_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _pdf_text(content):
    parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return " ".join(" ".join(parts).split())


def extract_from_link(url):
    if not url or not isinstance(url, str):
        return "Link broken"
    try:
        r = requests.get(url, headers=_UA, timeout=25)
        if r.status_code != 200:
            return "Link broken"
        ctype = r.headers.get("Content-Type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = _pdf_text(r.content)
        elif "html" in ctype or "text" in ctype:
            text = _html_text(r.text)
        else:
            return "Link broken"
        text = text.strip()
        return text if text else "Link broken"
    except Exception:
        return "Link broken"


def load_excel(file, sheet):
    wb = load_workbook(file, data_only=False)
    ws = wb[sheet]

    headers = [cell.value for cell in ws[1]]
    cleaned_headers = []
    for h in headers:
        if h is None or str(h).strip() == "":
            st.error("A column header is blank. Fix the header row and upload again.")
            st.stop()
        cleaned_headers.append(str(h).strip())

    if len(cleaned_headers) != len(set(cleaned_headers)):
        st.error("Two columns share the same header. Rename them and upload again.")
        st.stop()

    rows = [[cell.value for cell in row] for row in ws.iter_rows(min_row=2)]
    df = pd.DataFrame(rows, columns=cleaned_headers)

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


def chunk_bounds(total, chunks=25):
    """Split a row count into roughly equal slices so a progress bar can move."""
    size = max(1, -(-total // chunks))
    return [(i, min(i + size, total)) for i in range(0, total, size)]


# ==========================================
# STEP FUNCTIONS
# ==========================================
def run_extract(df, media_col, extract_col, allowed_types):
    if "Headline_Link" not in df.columns:
        raise ValueError("This file has no hyperlinks on the Headline column.")

    allowed_lower = {a.lower() for a in allowed_types}
    targets = [
        i for i in df.index
        if str(df.at[i, media_col]).strip().lower() in allowed_lower
        and (pd.isnull(df.at[i, extract_col]) or str(df.at[i, extract_col]).strip() == "")
    ]

    bar = st.progress(0, text="Extracting — 0%")
    broken = 0
    for n, i in enumerate(targets):
        result = extract_from_link(df.at[i, "Headline_Link"])
        df.at[i, extract_col] = result
        if result == "Link broken":
            broken += 1
        pct = int(((n + 1) / max(1, len(targets))) * 100)
        bar.progress(pct, text=f"Extracting — {pct}% ({n + 1} of {len(targets)})")
    bar.empty()
    return df, f"Processed {len(targets)} rows — {len(targets) - broken} extracted, {broken} link broken"


def run_combine(df, cols):
    if not cols:
        raise ValueError("Pick at least one column to combine.")

    total = len(df)
    bar = st.progress(0, text="Combining — 0%")
    pieces = []
    bounds = chunk_bounds(total)
    for n, (start, end) in enumerate(bounds):
        block = df.iloc[start:end][cols].fillna("").astype(str).agg(" ".join, axis=1)
        pieces.append(block.map(clean_text))
        pct = int(((n + 1) / len(bounds)) * 100)
        bar.progress(pct, text=f"Combining — {pct}%")
    df["Combined"] = pd.concat(pieces) if pieces else ""
    bar.empty()
    return df, f"Combined {len(cols)} column(s) across {total} rows"


def run_dedupe(df, exclude_cols):
    before = len(df)
    bar = st.progress(0, text="Comparing rows — 0%")
    check_cols = [c for c in df.columns if c not in exclude_cols]
    bar.progress(50, text="Comparing rows — 50%")
    df = df.drop_duplicates(subset=check_cols)
    bar.progress(100, text="Comparing rows — 100%")
    bar.empty()
    removed = before - len(df)
    return df, f"Removed {removed} duplicate row(s) — {len(df)} remaining"


def run_translate(df):
    if "Combined" not in df.columns:
        raise ValueError("Run 'Combine columns' first.")

    translator = GoogleTranslator(source="auto", target="en")
    texts = df["Combined"].astype(str).tolist()
    total = len(texts)
    bar = st.progress(0, text="Translating — 0%")
    out, failed = [], 0
    for n, t in enumerate(texts):
        try:
            out.append(translator.translate(t[:2000]))
        except Exception:
            out.append(t)
            failed += 1
        pct = int(((n + 1) / max(1, total)) * 100)
        bar.progress(pct, text=f"Translating — {pct}% ({n + 1} of {total})")
    df["Translated"] = out
    bar.empty()
    note = f"Translated {total - failed} of {total} rows"
    if failed:
        note += f" — {failed} kept in the original language"
    return df, note


def run_cluster(df, threshold):
    if "Combined" not in df.columns:
        raise ValueError("Run 'Combine columns' first.")

    model = load_model()
    texts = df["Combined"].astype(str).tolist()
    total = len(texts)
    batch_size = max(1, total // 30) if total > 30 else 1

    bar = st.progress(0, text="Generating embeddings — 0%")
    embeddings = []
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(model.encode(batch, convert_to_numpy=True))
        pct = min(100, int(((i + len(batch)) / total) * 100))
        bar.progress(pct, text=f"Generating embeddings — {pct}%")

    bar.progress(100, text="Grouping articles")
    emb = normalize(np.array(embeddings))
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    )
    df["Cluster"] = clustering.fit_predict(emb)
    summary = {
        c: " | ".join(df[df["Cluster"] == c]["Combined"].head(3).astype(str).tolist())
        for c in df["Cluster"].unique()
    }
    df["Cluster_Description"] = df["Cluster"].map(summary)
    bar.empty()
    return df, f"Found {df['Cluster'].nunique()} clusters across {total} rows"


def execute(step_key, fn, *args):
    """Run a step, handle undo, status and messaging in one place."""
    label = dict(STEPS)[step_key]
    save_undo(label)
    set_status(step_key, "Running")
    try:
        new_df, note = fn(st.session_state.data, *args)
        st.session_state.data = new_df
        st.session_state.export_buffer = None
        set_status(step_key, "Done")
        st.success(note)
        return True
    except Exception as e:
        set_status(step_key, "Error")
        st.error(str(e))
        return False


# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="app-header">
    <div class="app-header-title">Insights Copilot</div>
    <div class="app-header-sub">Media intelligence pipeline · cleaning, translation and clustering</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR — data source and settings
# ==========================================
with st.sidebar:
    st.markdown('<div class="section-label">Data source</div>', unsafe_allow_html=True)
    file = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])

    if file is not None:
        sheet = st.selectbox("Sheet", pd.ExcelFile(file).sheet_names)
        if st.button("Load data", type="primary", use_container_width=True):
            st.session_state.data = load_excel(file, sheet)
            st.session_state.status = {k: "Not run" for k, _ in STEPS}
            st.session_state.undo = None
            st.session_state.export_buffer = None
            st.rerun()

    if st.session_state.data is not None:
        df_side = st.session_state.data

        st.markdown("---")
        st.markdown('<div class="section-label">Step settings</div>', unsafe_allow_html=True)

        with st.expander("Extract text from links", expanded=False):
            media_col = st.selectbox(
                "Media type column", df_side.columns,
                index=list(df_side.columns).index("Media Type") if "Media Type" in df_side.columns else 0
            )
            extract_col = st.selectbox(
                "Extract text column", df_side.columns,
                index=list(df_side.columns).index("Extract Text") if "Extract Text" in df_side.columns else 0
            )
            allowed_types = st.multiselect(
                "Media types to process",
                ["Online", "Newspaper", "TV", "Radio"],
                default=["Online", "Newspaper"]
            )

        with st.expander("Combine columns", expanded=False):
            combine_cols = st.multiselect("Columns to combine", df_side.columns)

        with st.expander("Remove duplicates", expanded=False):
            exclude_cols = st.multiselect("Columns to ignore when comparing", df_side.columns)

        with st.expander("Semantic clustering", expanded=False):
            threshold = st.slider("Distance threshold (lower is stricter)", 0.25, 0.35, 0.28, step=0.01)

        st.markdown("---")
        st.markdown('<div class="section-label">Run everything</div>', unsafe_allow_html=True)
        run_all_steps = st.multiselect(
            "Steps to include",
            [name for _, name in STEPS],
            default=[name for _, name in STEPS],
        )
        run_all = st.button("Run selected steps", type="primary", use_container_width=True)

        st.markdown("---")
        undo_label = st.session_state.undo[0] if st.session_state.undo else None
        if st.button(
            f"Undo: {undo_label}" if undo_label else "Undo last step",
            disabled=undo_label is None,
            use_container_width=True,
        ):
            _, prev_df, prev_status = st.session_state.undo
            st.session_state.data = prev_df
            st.session_state.status = prev_status
            st.session_state.undo = None
            st.session_state.export_buffer = None
            st.rerun()

        if st.button("Start over", use_container_width=True):
            st.session_state.data = None
            st.session_state.status = {k: "Not run" for k, _ in STEPS}
            st.session_state.undo = None
            st.session_state.export_buffer = None
            st.rerun()
    else:
        run_all = False
        run_all_steps = []

df = st.session_state.data

if df is None:
    st.info("Upload an Excel file in the sidebar to start.")
    st.stop()

# ==========================================
# PIPELINE STRIP + COUNTS + PREVIEW
# ==========================================
strip = "".join(
    '<div class="pipeline-step %s"></div>' %
    {"Done": "done", "Running": "active", "Error": "error"}.get(get_status(k), "")
    for k, _ in STEPS
)
done_count = sum(1 for k, _ in STEPS if get_status(k) == "Done")
st.markdown(
    f'<div class="pipeline-progress">{strip}</div>'
    f'<div class="pipeline-caption">{done_count} of {len(STEPS)} steps done</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", len([c for c in df.columns if c != "Headline_Link"]))
m3.metric("Steps done", f"{done_count} / {len(STEPS)}")

st.markdown('<div class="section-label" style="margin-top:18px;">Preview</div>', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Steps</div>', unsafe_allow_html=True)

# ==========================================
# RUN ALL
# ==========================================
if run_all:
    name_to_key = {name: key for key, name in STEPS}
    for name in [n for _, n in STEPS if n in run_all_steps]:
        key = name_to_key[name]
        st.markdown(f"**{name}**")
        if key == "step1":
            ok = execute(key, run_extract, media_col, extract_col, allowed_types)
        elif key == "step2":
            ok = execute(key, run_combine, combine_cols)
        elif key == "step3":
            ok = execute(key, run_dedupe, exclude_cols)
        elif key == "step4":
            ok = execute(key, run_translate)
        else:
            ok = execute(key, run_cluster, threshold)
        if not ok:
            st.warning("Stopped here. Fix the setting above and run again.")
            break
    df = st.session_state.data

# ==========================================
# INDIVIDUAL STEPS
# ==========================================
def step_header(key):
    name = dict(STEPS)[key]
    mark = {"Done": "\u2713 ", "Error": "\u2715 "}.get(get_status(key), "")
    return f"{mark}{name}"


with st.expander(step_header("step1"), expanded=get_status("step1") != "Done"):
    st.markdown(
        '<p class="step-note">Fills blank Extract Text cells by pulling the text behind the Headline link. '
        'Unreachable or non-text links are marked "Link broken".</p>',
        unsafe_allow_html=True
    )
    if st.button("Run", key="run1"):
        execute("step1", run_extract, media_col, extract_col, allowed_types)
    st.markdown(status_pill("step1"), unsafe_allow_html=True)

with st.expander(step_header("step2"), expanded=get_status("step2") != "Done"):
    st.markdown(
        '<p class="step-note">Joins the selected columns into one Combined field. '
        'Translation and clustering both read from it.</p>',
        unsafe_allow_html=True
    )
    if st.button("Run", key="run2"):
        execute("step2", run_combine, combine_cols)
    st.markdown(status_pill("step2"), unsafe_allow_html=True)

with st.expander(step_header("step3"), expanded=get_status("step3") != "Done"):
    st.markdown(
        '<p class="step-note">Drops rows that match on every column apart from the ones you excluded.</p>',
        unsafe_allow_html=True
    )
    if st.button("Run", key="run3"):
        execute("step3", run_dedupe, exclude_cols)
    st.markdown(status_pill("step3"), unsafe_allow_html=True)

with st.expander(step_header("step4"), expanded=get_status("step4") != "Done"):
    st.markdown(
        '<p class="step-note">Translates the Combined field to English with Google Translate, '
        'detecting the source language automatically.</p>',
        unsafe_allow_html=True
    )
    if st.button("Run", key="run4"):
        execute("step4", run_translate)
    st.markdown(status_pill("step4"), unsafe_allow_html=True)

with st.expander(step_header("step5"), expanded=get_status("step5") != "Done"):
    st.markdown(
        '<p class="step-note">Groups similar articles using multilingual sentence embeddings and '
        'agglomerative clustering on cosine similarity.</p>',
        unsafe_allow_html=True
    )
    if st.button("Run", key="run5"):
        execute("step5", run_cluster, threshold)
    st.markdown(status_pill("step5"), unsafe_allow_html=True)

# ==========================================
# EXPORT
# ==========================================
st.markdown("---")
st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)

e1, e2 = st.columns([2, 1])
with e1:
    filename = st.text_input("File name", "output.xlsx", label_visibility="collapsed")
with e2:
    if st.button("Prepare file", use_container_width=True):
        st.session_state.export_buffer = to_excel(st.session_state.data).getvalue()

if st.session_state.export_buffer is not None:
    st.download_button(
        "Download Excel",
        data=st.session_state.export_buffer,
        file_name=filename,
    )
else:
    st.caption("Prepare the file first, then the download button appears here.")

import streamlit as st
import pandas as pd
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
    page_title="Notion Data Pipeline",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# STATE ENGINE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "steps" not in st.session_state:
    st.session_state.steps = {
        "1": False,
        "2": False,
        "3": False,
        "4": False,
        "5": False,
        "6": False
    }

def done(step):
    return st.session_state.steps[step]

def mark(step):
    st.session_state.steps[step] = True

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
def preview(df, title):
    st.markdown(f"### 👀 {title}")
    st.dataframe(df.head(10), use_container_width=True)

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    return df

def remove_duplicates(df, exclude):
    subset = [c for c in df.columns if c not in exclude]
    return df.drop_duplicates(subset=subset)

def extract_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return " ".join([s for s in sentences if any(k.lower() in s.lower() for k in keywords)])

def translate(df):
    tr = GoogleTranslator(source='auto', target='en')
    df["Translated"] = df["Combined"].apply(lambda x: tr.translate(str(x)[:2000]))
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
    return df

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

# ==========================================
# SIDEBAR (NOTION STYLE NAV)
# ==========================================
st.sidebar.title("📊 Pipeline")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "🧩 Step 1 - Combine",
        "🧹 Step 2 - Dedup",
        "🔑 Step 3 - Keywords",
        "🌍 Step 4 - Translate",
        "💬 Step 5 - Sentiment",
        "📦 Step 6 - Cluster",
        "📥 Output"
    ]
)

file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

# ==========================================
# LOAD DATA
# ==========================================
if file and st.session_state.data is None:
    xls = pd.ExcelFile(file)
    sheet = st.sidebar.selectbox("Sheet", xls.sheet_names)
    st.session_state.data = pd.read_excel(file, sheet_name=sheet)

df = st.session_state.data

if df is None:
    st.title("📊 Notion Data Pipeline")
    st.info("Upload file to begin")
    st.stop()

# ==========================================
# OVERVIEW PAGE
# ==========================================
if page == "📊 Overview":

    st.title("📊 Pipeline Overview")

    st.write("### Progress")

    for k, v in st.session_state.steps.items():
        st.write(f"Step {k}: {'🟢 Done' if v else '🟡 Pending'}")

    preview(df, "Raw Data")

# ==========================================
# STEP 1
# ==========================================
if page == "🧩 Step 1 - Combine":

    st.title("Step 1 — Combine Columns")

    cols = st.multiselect("Select columns", df.columns)

    if st.button("Run Step 1"):

        df = create_combined(df, cols)
        st.session_state.data = df
        mark("1")

    preview(df, "After Step 1" if done("1") else "Not Run")

# ==========================================
# STEP 2
# ==========================================
if page == "🧹 Step 2 - Dedup":

    st.title("Step 2 — Remove Duplicates")

    if not done("1"):
        st.warning("Complete Step 1 first")
        st.stop()

    exclude = st.multiselect("Exclude columns", df.columns)

    if st.button("Run Step 2"):
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        mark("2")

    preview(df, "After Step 2" if done("2") else "Not Run")

# ==========================================
# STEP 3
# ==========================================
if page == "🔑 Step 3 - Keywords":

    st.title("Step 3 — Keyword Matching")

    if not done("1"):
        st.warning("Complete Step 1 first")
        st.stop()

    kw = st.text_input("Keywords (comma separated)")
    tag_col = st.text_input("Tag column", "Tags")

    if st.button("Run Step 3"):

        keywords = [k.strip() for k in kw.split(",") if k.strip()]

        df[tag_col] = df["Combined"].apply(
            lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
        )

        st.session_state.data = df
        mark("3")

    preview(df, "After Step 3" if done("3") else "Not Run")

# ==========================================
# STEP 4
# ==========================================
if page == "🌍 Step 4 - Translate":

    st.title("Step 4 — Translation")

    if st.button("Run Step 4"):

        df = translate(df)
        st.session_state.data = df
        mark("4")

    preview(df, "After Step 4" if done("4") else "Not Run")

# ==========================================
# STEP 5
# ==========================================
if page == "💬 Step 5 - Sentiment":

    st.title("Step 5 — Sentiment")

    source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])

    if st.button("Run Step 5"):

        model = load_sentiment()

        df["Sentiment"] = df[source].apply(
            lambda x: model(str(x)[:512])[0]["label"]
        )

        st.session_state.data = df
        mark("5")

    preview(df, "After Step 5" if done("5") else "Not Run")

# ==========================================
# STEP 6
# ==========================================
if page == "📦 Step 6 - Cluster":

    st.title("Step 6 — Clustering")

    threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

    if st.button("Run Step 6"):

        df = cluster(df, threshold)
        st.session_state.data = df
        mark("6")

    preview(df, "After Step 6" if done("6") else "Not Run")

# ==========================================
# OUTPUT
# ==========================================
if page == "📥 Output":

    st.title("Final Output")

    st.dataframe(df, use_container_width=True)

    name = st.text_input("File name", "output.xlsx")

    st.download_button(
        "Download Excel",
        data=to_excel(df),
        file_name=name
    )
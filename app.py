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
    page_title="Data Cleaner Pro",
    layout="wide",
    page_icon="📊"
)

# ==========================================
# STATE ENGINE (NEW)
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

if "completed" not in st.session_state:
    st.session_state.completed = set()

def mark_done(step):
    st.session_state.completed.add(step)

def is_done(step):
    return step in st.session_state.completed

def can_run(step, required=None):
    if required is None:
        return True
    return all(r in st.session_state.completed for r in required)

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

def preview(df, title):
    st.markdown(f"### 👀 {title}")
    st.dataframe(df.head(10), use_container_width=True)

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, exclude):
    subset = [c for c in df.columns if c not in exclude]
    return df.drop_duplicates(subset=subset)

def extract_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return " ".join([s for s in sentences if any(k.lower() in s.lower() for k in keywords)])

def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    df["Translated"] = df["Combined"].apply(
        lambda x: translator.translate(str(x)[:2000])
    )
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
# LOAD DATA
# ==========================================
st.title("📊 Data Cleaner Pro")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True

df = st.session_state.data

if df is None:
    st.stop()

# ==========================================
# STEP 1 — REQUIRED
# ==========================================
st.header("🧩 Step 1 — Combine Columns (Required)")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Step 1"):

    if len(cols) == 0:
        st.error("Select at least one column")
    else:
        df = create_combined(df, cols)
        st.session_state.data = df
        mark_done("step1")
        st.success("Step 1 Done ✔")

# ONLY preview AFTER run
if is_done("step1"):
    preview(df, "After Step 1")

# BLOCK next steps if not done
if not is_done("step1"):
    st.warning("⚠ Please complete Step 1 first")
    st.stop()

# ==========================================
# STEP 2 — DEDUP
# ==========================================
st.header("🧹 Step 2 — Remove Duplicates")

exclude = st.multiselect("Exclude columns", df.columns)

if st.button("▶ Run Step 2"):

    df = remove_duplicates(df, exclude)
    st.session_state.data = df
    mark_done("step2")
    st.success("Step 2 Done ✔")

if is_done("step2"):
    preview(df, "After Step 2")

# ==========================================
# STEP 3 — KEYWORDS
# ==========================================
st.header("🔑 Step 3 — Keyword Matching")

num = st.number_input("Groups", 1, 10, 1)

if st.button("▶ Run Step 3"):

    for i in range(num):

        kw_text = st.text_input("Keywords", key=f"k{i}")
        tag_col = st.text_input("Tag column", f"Tags_{i+1}")

        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]

        df[tag_col] = df["Combined"].apply(
            lambda x: ", ".join([k for k in keywords if k.lower() in str(x).lower()])
        )

    st.session_state.data = df
    mark_done("step3")
    st.success("Step 3 Done ✔")

if is_done("step3"):
    preview(df, "After Step 3")

# ==========================================
# STEP 4 — TRANSLATION
# ==========================================
st.header("🌍 Step 4 — Translation")

if st.button("▶ Run Step 4"):

    df = translate(df)
    st.session_state.data = df
    mark_done("step4")
    st.success("Step 4 Done ✔")

if is_done("step4"):
    preview(df, "After Step 4")

# ==========================================
# STEP 5 — SENTIMENT
# ==========================================
st.header("💬 Step 5 — Sentiment")

source = st.radio("Source", ["Combined", "Translated"] if "Translated" in df.columns else ["Combined"])

if st.button("▶ Run Step 5"):

    model = load_sentiment_model()

    results = []

    for _, row in df.iterrows():

        text = str(row[source])

        if "Combined" in df.columns:
            results.append(model(text[:512])[0]["label"])
        else:
            results.append("NO_MENTION")

    df["Sentiment"] = results
    st.session_state.data = df
    mark_done("step5")
    st.success("Step 5 Done ✔")

if is_done("step5"):
    preview(df, "After Step 5")

# ==========================================
# STEP 6 — CLUSTERING
# ==========================================
st.header("📦 Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

if st.button("▶ Run Step 6"):

    df = cluster(df, threshold)
    st.session_state.data = df
    mark_done("step6")
    st.success("Step 6 Done ✔")

if is_done("step6"):
    preview(df, "After Step 6")

# ==========================================
# FINAL OUTPUT
# ==========================================
st.markdown("---")
st.subheader("📦 Final Output")

preview(df, "Final Data")

st.download_button(
    "📥 Download Excel",
    data=to_excel(df),
    file_name="output.xlsx"
)
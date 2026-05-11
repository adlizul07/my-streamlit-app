import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from transformers import pipeline

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Your Number 1 Data Cleaner!", layout="wide")
DetectorFactory.seed = 0

# ==========================================
# STATUS TRACKER
# ==========================================
def status_icon(step):
    return st.session_state.get(step, "🟡 Not Run")

def set_status(step, value):
    st.session_state[step] = value

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
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )

# ==========================================
# HELPERS
# ==========================================
def clean_text(text):
    if pd.isnull(text):
        return text
    return str(text).replace('_x000D_', ' ').replace('\r', ' ').replace('\n', ' ').strip()

def create_combined(df, cols):
    df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["Combined"] = df["Combined"].apply(clean_text)
    return df

def remove_duplicates(df, exclude):
    subset = [c for c in df.columns if c not in exclude]
    return df.drop_duplicates(subset=subset) if subset else df

def parse_keywords(text):
    if not text:
        return []
    return [k.strip() for k in text.split(",") if k.strip()]

def extract_sentences(text, keywords):
    if pd.isnull(text):
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', str(text))

    matched = []
    for s in sentences:
        if any(k.lower() in s.lower() for k in keywords):
            matched.append(s.strip())

    return "\n".join(matched)

# ==========================================
# BRAND HELPERS
# ==========================================
def get_brand_aliases(brand):

    if pd.isnull(brand):
        return []

    brand = str(brand).lower()

    aliases = set([brand])

    cleaned = re.sub(r'\b(berhad|bhd|sdn bhd|ltd|inc|corp)\b', '', brand).strip()
    aliases.add(cleaned)

    words = cleaned.split()
    if len(words) > 1:
        aliases.add("".join([w[0] for w in words if w]))

    manual = {
        "maybank": ["malayan banking", "m2u"],
        "tm": ["telekom malaysia"],
        "pr1ma": ["perbadanan pr1ma malaysia", "prima"],
        "kwsp": ["epf", "employees provident fund"]
    }

    if brand in manual:
        aliases.update(manual[brand])

    return list(aliases)


def extract_brand_sentence(text, aliases):

    if pd.isnull(text):
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', str(text))

    return " ".join([s for s in sentences if any(a in s.lower() for a in aliases)])

# ==========================================
# KEYWORD ENGINE
# ==========================================
def keyword_match_v2(df, kw_file, kw_text, tag_col, sentence_col, create_sentence):

    keywords = []

    if kw_file:
        kw_df = pd.read_excel(kw_file)
        keywords = kw_df.iloc[:, 0].dropna().astype(str).tolist()

    elif kw_text:
        keywords = parse_keywords(kw_text)

    def find_tags(text):
        if pd.isnull(text):
            return ""
        text_low = str(text).lower()
        return ", ".join(sorted(set([k for k in keywords if k.lower() in text_low])))

    df[tag_col] = df["Combined"].apply(find_tags)

    if create_sentence:
        df[sentence_col] = df["Combined"].apply(lambda x: extract_sentences(x, keywords))

    return df

# ==========================================
# TRANSLATION
# ==========================================
def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    def tr(x):
        try:
            return translator.translate(str(x)[:2000])
        except:
            return x

    df["Translated"] = df["Combined"].apply(tr)
    df["Translated"] = df["Translated"].apply(clean_text)
    return df

# ==========================================
# CLUSTERING
# ==========================================
def cluster(df, threshold):
    model = load_model()

    texts = df["Combined"].fillna("").astype(str).tolist()
    emb = model.encode(texts, convert_to_numpy=True)
    emb = normalize(emb)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    df["Cluster"] = clustering.fit_predict(emb)
    return df

# ==========================================
# EXPORT
# ==========================================
def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

# ==========================================
# UI
# ==========================================
st.title("📊 Your Number 1 Data Cleaner!")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

# =========================
# LOAD FILE STATUS
# =========================
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True
    set_status("step_file", "🟢 Loaded")

df = st.session_state.data

if df is None:
    st.stop()

st.subheader(f"Step File Load: {status_icon('step_file')}")
st.dataframe(df.head())

# ==========================================
# STEP 1 — COMBINED
# ==========================================
st.header("Step 1 — Combined Column")

if st.button("Skip Step 1"):
    set_status("step1", "🟡 Skipped")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Combined"):
    try:
        set_status("step1", "🔵 Running")
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("step1", "🟢 Done")
    except:
        set_status("step1", "🔴 Error")

st.write("Status:", status_icon("step1"))

# ==========================================
# STEP 2 — DUPLICATES
# ==========================================
st.header("Step 2 — Remove Duplicates")

if st.button("Skip Step 2"):
    set_status("step2", "🟡 Skipped")

exclude = st.multiselect("Exclude columns", df.columns)

if st.button("▶ Run Duplicates"):
    try:
        set_status("step2", "🔵 Running")
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        set_status("step2", "🟢 Done")
    except:
        set_status("step2", "🔴 Error")

st.write("Status:", status_icon("step2"))

# ==========================================
# STEP 3 — KEYWORDS
# ==========================================
st.header("Step 3 — Keyword Matching")

st.info("Or you can add the keyword manually below 👇")

num_groups = st.number_input("How many keyword groups?", 1, 10, 1)

for i in range(num_groups):

    kw_file = st.file_uploader(f"Keyword file {i+1}", type=["xlsx"], key=f"kw{i}")
    kw_text = st.text_input(f"OR manual keywords (comma separated)", key=f"kwt{i}")

    tag_col = st.text_input(f"Tag column {i+1}", f"Tags_{i+1}")
    sent_col = st.text_input(f"Sentence column {i+1}", f"Sent_{i+1}")

    create_sentence = st.checkbox("Create sentence?", key=f"cs{i}")

    if st.button(f"Run Group {i+1}"):

        try:
            set_status("step3", "🔵 Running")
            df = keyword_match_v2(df, kw_file, kw_text, tag_col, sent_col, create_sentence)
            st.session_state.data = df
            set_status("step3", "🟢 Done")
        except:
            set_status("step3", "🔴 Error")

st.write("Status:", status_icon("step3"))

# ==========================================
# STEP 4 — TRANSLATION
# ==========================================
st.header("Step 4 — Translation")

if st.button("Skip Step 4"):
    set_status("step4", "🟡 Skipped")

if st.button("▶ Run Translation"):
    try:
        set_status("step4", "🔵 Running")
        df = translate(df)
        st.session_state.data = df
        set_status("step4", "🟢 Done")
    except:
        set_status("step4", "🔴 Error")

st.write("Status:", status_icon("step4"))

# ==========================================
# STEP 5 — SENTIMENT
# ==========================================
st.header("Step 5 — Sentiment Analysis")

sent_source = st.radio("Use:", ["Combined", "Translated"])
brand_col = st.selectbox("Brand column", df.columns)

if st.button("▶ Run Sentiment"):
    try:
        set_status("step5", "🔵 Running")

        model = load_sentiment_model()

        sentiments, scores, sentences = [], [], []

        for _, row in df.iterrows():

            text = row[sent_source]
            brand = row[brand_col]

            aliases = get_brand_aliases(brand)
            brand_text = extract_brand_sentence(text, aliases)

            if brand_text == "":
                sentiments.append("NO_MENTION")
                scores.append(None)
                sentences.append("")
                continue

            result = model(brand_text[:512])[0]

            sentiments.append(result["label"])
            scores.append(result["score"])
            sentences.append(brand_text)

        df["Sentiment"] = sentiments
        df["Score"] = scores
        df["Sentence"] = sentences

        st.session_state.data = df
        set_status("step5", "🟢 Done")

    except:
        set_status("step5", "🔴 Error")

st.write("Status:", status_icon("step5"))

# ==========================================
# STEP 6 — CLUSTERING
# ==========================================
st.header("Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

if st.button("▶ Run Clustering"):
    try:
        set_status("step6", "🔵 Running")
        df = cluster(df, threshold)
        st.session_state.data = df
        set_status("step6", "🟢 Done")
    except:
        set_status("step6", "🔴 Error")

st.write("Status:", status_icon("step6"))

# ==========================================
# OUTPUT
# ==========================================
st.header("Final Output")
st.dataframe(st.session_state.data)

st.download_button(
    "📥 Download",
    data=to_excel(st.session_state.data),
    file_name="output.xlsx"
)
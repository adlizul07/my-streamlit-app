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
# STATUS SYSTEM
# ==========================================
def status_icon(step):
    return st.session_state.get(step, "🟡 Not Run")

def set_status(step, value):
    st.session_state[step] = value

# ==========================================
# STEP SKIP CONTROL
# ==========================================
for i in range(1, 7):
    if f"skip_step{i}" not in st.session_state:
        st.session_state[f"skip_step{i}"] = False

def set_skip(step):
    st.session_state[f"skip_step{step}"] = True

def is_skipped(step):
    return st.session_state[f"skip_step{step}"]

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
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

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
    return "\n".join([s for s in sentences if any(k.lower() in s.lower() for k in keywords)])

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
# PROCESS FUNCTIONS
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

def translate(df):
    translator = GoogleTranslator(source='auto', target='en')

    def tr(x):
        try:
            return translator.translate(str(x)[:2000])
        except:
            return x

    df["Translated"] = df["Combined"].apply(tr)
    return df

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

# LOAD FILE
if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)
    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True
    set_status("step_file", "🟢 Loaded")

df = st.session_state.data

if df is None:
    st.stop()

st.subheader(f"File Status: {status_icon('step_file')}")
st.dataframe(df.head())

# ==========================================
# STEP 1
# ==========================================
st.header("Step 1 — Combined Column")

cols = st.multiselect("Select columns", df.columns)

c1, c2 = st.columns(2)
run1 = c1.button("▶ Run", disabled=is_skipped(1))
skip1 = c2.button("⏭ Skip")

if skip1:
    set_skip(1)
    set_status("step1", "🟡 Skipped")

if run1 and not is_skipped(1):
    try:
        set_status("step1", "🔵 Running")
        df = create_combined(df, cols)
        st.session_state.data = df
        set_status("step1", "🟢 Done")
    except:
        set_status("step1", "🔴 Error")

st.write(status_icon("step1"))

# ==========================================
# STEP 2
# ==========================================
st.header("Step 2 — Remove Duplicates")

exclude = st.multiselect("Exclude columns", df.columns)

c1, c2 = st.columns(2)
run2 = c1.button("▶ Run", disabled=is_skipped(2))
skip2 = c2.button("⏭ Skip")

if skip2:
    set_skip(2)
    set_status("step2", "🟡 Skipped")

if run2 and not is_skipped(2):
    try:
        set_status("step2", "🔵 Running")
        df = remove_duplicates(df, exclude)
        st.session_state.data = df
        set_status("step2", "🟢 Done")
    except:
        set_status("step2", "🔴 Error")

st.write(status_icon("step2"))

# ==========================================
# STEP 3
# ==========================================
st.header("Step 3 — Keyword Matching")

num_groups = st.number_input("Keyword groups", 1, 10, 1)

for i in range(num_groups):

    st.subheader(f"Group {i+1}")

    kw_file = st.file_uploader("Keyword file", type=["xlsx"], key=f"kw{i}")
    kw_text = st.text_input("Manual keywords", key=f"kwt{i}")

    tag_col = st.text_input("Tag column", f"Tags_{i+1}")
    sent_col = st.text_input("Sentence column", f"Sent_{i+1}")
    create_sentence = st.checkbox("Create sentence", key=f"cs{i}")

    c1, c2 = st.columns(2)
    run3 = c1.button("▶ Run", key=f"run{i}", disabled=is_skipped(3))
    skip3 = c2.button("⏭ Skip", key=f"skip{i}")

    if skip3:
        set_skip(3)
        set_status("step3", "🟡 Skipped")

    if run3 and not is_skipped(3):
        try:
            set_status("step3", "🔵 Running")
            df = keyword_match_v2(df, kw_file, kw_text, tag_col, sent_col, create_sentence)
            st.session_state.data = df
            set_status("step3", "🟢 Done")
        except:
            set_status("step3", "🔴 Error")

st.write(status_icon("step3"))

# ==========================================
# STEP 4
# ==========================================
st.header("Step 4 — Translation")

c1, c2 = st.columns(2)
run4 = c1.button("▶ Run", disabled=is_skipped(4))
skip4 = c2.button("⏭ Skip")

if skip4:
    set_skip(4)
    set_status("step4", "🟡 Skipped")

if run4 and not is_skipped(4):
    try:
        set_status("step4", "🔵 Running")
        df = translate(df)
        st.session_state.data = df
        set_status("step4", "🟢 Done")
    except:
        set_status("step4", "🔴 Error")

st.write(status_icon("step4"))

# ==========================================
# STEP 5 — SENTIMENT
# ==========================================
st.header("Step 5 — Sentiment")

sent_source = st.radio("Source", ["Combined", "Translated"])
brand_col = st.selectbox("Brand column", df.columns)

c1, c2 = st.columns(2)
run5 = c1.button("▶ Run", disabled=is_skipped(5))
skip5 = c2.button("⏭ Skip")

if skip5:
    set_skip(5)
    set_status("step5", "🟡 Skipped")

if run5 and not is_skipped(5):
    try:
        set_status("step5", "🔵 Running")

        model = load_sentiment_model()

        sentiments, scores = [], []

        for _, row in df.iterrows():

            text = row[sent_source]
            brand = row[brand_col]

            aliases = get_brand_aliases(brand)
            brand_text = extract_brand_sentence(text, aliases)

            if not brand_text:
                sentiments.append("NO_MENTION")
                scores.append(None)
                continue

            result = model(brand_text[:512])[0]
            sentiments.append(result["label"])
            scores.append(result["score"])

        df["Sentiment"] = sentiments
        df["Score"] = scores

        st.session_state.data = df
        set_status("step5", "🟢 Done")

    except:
        set_status("step5", "🔴 Error")

st.write(status_icon("step5"))

# ==========================================
# STEP 6
# ==========================================
st.header("Step 6 — Clustering")

threshold = st.slider("Strictness", 0.25, 0.35, 0.28)

c1, c2 = st.columns(2)
run6 = c1.button("▶ Run", disabled=is_skipped(6))
skip6 = c2.button("⏭ Skip")

if skip6:
    set_skip(6)
    set_status("step6", "🟡 Skipped")

if run6 and not is_skipped(6):
    try:
        set_status("step6", "🔵 Running")
        df = cluster(df, threshold)
        st.session_state.data = df
        set_status("step6", "🟢 Done")
    except:
        set_status("step6", "🔴 Error")

st.write(status_icon("step6"))

# ==========================================
# OUTPUT
# ==========================================
st.header("Final Output")

st.dataframe(st.session_state.data)

st.download_button(
    "📥 Download Excel",
    data=to_excel(st.session_state.data),
    file_name="output.xlsx"
)
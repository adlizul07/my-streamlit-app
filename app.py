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

    aliases = set()
    aliases.add(brand)

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

    matched = []

    for s in sentences:
        sl = s.lower()
        if any(a in sl for a in aliases):
            matched.append(s)

    return " ".join(matched)

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
        matched = [k for k in keywords if k.lower() in text_low]
        return ", ".join(sorted(set(matched)))

    df[tag_col] = df["Combined"].apply(find_tags)

    if create_sentence:
        df[sentence_col] = df["Combined"].apply(
            lambda x: extract_sentences(x, keywords)
        )

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
    emb = model.encode(texts, convert_to_numpy=True, batch_size=32)
    emb = normalize(emb)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    labels = clustering.fit_predict(emb)
    df["Cluster"] = labels

    cluster_desc = {}

    for c in np.unique(labels):
        sample = df.loc[df["Cluster"] == c, "Combined"].iloc[0]
        cluster_desc[c] = str(sample)[:120] + "..."

    df["Cluster Description"] = df["Cluster"].map(cluster_desc)

    return df

# ==========================================
# EXPORT
# ==========================================
def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    buffer.seek(0)
    return buffer

# ==========================================
# UI
# ==========================================
st.title("📊 Your Number 1 Data Cleaner!")

file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file and not st.session_state.file_loaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Select Sheet", xls.sheet_names)

    st.session_state.data = pd.read_excel(file, sheet_name=sheet)
    st.session_state.file_loaded = True

df = st.session_state.data

if df is None:
    st.info("⬆️ Upload file to start")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(5))

# ==========================================
# STEP 1 — COMBINED
# ==========================================
st.header("Step 1 — Combined Column")

cols = st.multiselect("Select columns", df.columns)

if st.button("▶ Run Combined"):
    df = create_combined(df, cols)
    st.session_state.data = df
    st.success("✅ Combined created")

# ==========================================
# STEP 2 — DUPLICATES
# ==========================================
st.header("Step 2 — Remove Duplicates")

exclude = st.multiselect("Exclude columns", df.columns)

if st.button("▶ Run Duplicates"):
    df = remove_duplicates(df, exclude)
    st.session_state.data = df
    st.success("✅ Duplicates removed")

# ==========================================
# STEP 3 — KEYWORDS
# ==========================================
st.header("Step 3 — Keyword Matching")

num_groups = st.number_input("How many keyword groups?", 1, 10, 1)

for i in range(num_groups):

    kw_file = st.file_uploader(f"Keyword file {i+1}", type=["xlsx"], key=f"kwf{i}")
    kw_text = st.text_input(f"Keywords {i+1}", key=f"kwt{i}")

    tag_col = st.text_input(f"Tag column {i+1}", f"Keyword_Tags_{i+1}", key=f"tag{i}")
    sent_col = st.text_input(f"Sentence column {i+1}", f"Keyword_Sentences_{i+1}", key=f"sent{i}")

    create_sentence = st.checkbox(f"Create sentence {i+1}", key=f"chk{i}")

    if st.button(f"Run Group {i+1}", key=f"run{i}"):

        df = st.session_state.data

        df = keyword_match_v2(df, kw_file, kw_text, tag_col, sent_col, create_sentence)

        st.session_state.data = df
        st.success(f"Group {i+1} done")

# ==========================================
# STEP 4 — TRANSLATION
# ==========================================
st.header("Step 4 — Translation")

if st.button("▶ Run Translation"):
    df = st.session_state.data
    df = translate(df)
    st.session_state.data = df
    st.success("Translation done")

# ==========================================
# STEP 5 — SENTIMENT ANALYSIS (NEW)
# ==========================================
st.header("Step 5 — Sentiment Analysis (Brand Level)")

st.warning("Use 'Translated' for better accuracy if data is multilingual.")

sent_source = st.radio("Sentiment source:", ["Combined", "Translated"])

brand_col = st.selectbox("Select brand column", df.columns)

if st.button("▶ Run Sentiment Analysis"):

    df = st.session_state.data
    model = load_sentiment_model()

    sentiments = []
    scores = []
    sentences = []

    for _, row in df.iterrows():

        text = row[sent_source]
        brand = row[brand_col]

        aliases = get_brand_aliases(brand)
        brand_text = extract_brand_sentence(text, aliases)

        if brand_text.strip() == "":
            sentiments.append("NO_MENTION")
            scores.append(None)
            sentences.append("")
            continue

        try:
            result = model(brand_text[:512])[0]

            sentiments.append(result["label"])
            scores.append(result["score"])
            sentences.append(brand_text)

        except:
            sentiments.append("ERROR")
            scores.append(None)
            sentences.append("")

    df["Brand_Sentiment"] = sentiments
    df["Sentiment_Score"] = scores
    df["Matched_Sentence"] = sentences

    st.session_state.data = df
    st.success("Sentiment analysis completed")

# ==========================================
# STEP 6 — CLUSTERING
# ==========================================
st.header("Step 6 — Clustering")

threshold = st.slider("Strictness", 0.1, 1.0, 0.28)

if st.button("▶ Run Clustering"):
    df = st.session_state.data
    df = cluster(df, threshold)
    st.session_state.data = df
    st.success("Clustering done")

# ==========================================
# OUTPUT
# ==========================================
st.header("Final Output")

st.dataframe(st.session_state.data.head(5))

excel = to_excel(st.session_state.data)

st.download_button(
    "📥 Download Excel",
    data=excel,
    file_name="output.xlsx"
)
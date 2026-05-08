import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Media Cleaning & NLP Pipeline", layout="wide")
DetectorFactory.seed = 0

# ==========================================
# SESSION STATE
# ==========================================
if "data" not in st.session_state:
    st.session_state.data = None

# ==========================================
# MODEL
# ==========================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# ==========================================
# HELPERS
# ==========================================
def clean_text(text):
    if pd.isnull(text):
        return text
    return str(text).replace('_x000D_', ' ').replace('\r', ' ').replace('\n', ' ').strip()

def create_combined_column(df, cols):
    try:
        df["Combined"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
        df["Combined"] = df["Combined"].apply(clean_text)
        return df, True
    except Exception as e:
        st.error(f"❌ Failed to create Combined column: {e}")
        return df, False

def remove_duplicates(df, excluded):
    try:
        subset = [c for c in df.columns if c not in excluded]
        df = df.drop_duplicates(subset=subset)
        return df, True
    except Exception as e:
        st.error(f"❌ Duplicate removal failed: {e}")
        return df, False

def extract_sentences(text, keywords):
    if pd.isnull(text):
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return "\n".join(
        s for s in sentences
        if any(k.lower() in s.lower() for k in keywords)
    )

def keyword_match(df, keyword_df, col_name):
    try:
        keywords = keyword_df.iloc[:, 0].dropna().astype(str).tolist()
        df[col_name] = df["Combined"].apply(lambda x: extract_sentences(x, keywords))
        return df, True
    except Exception as e:
        st.error(f"❌ Keyword matching failed: {e}")
        return df, False

def translate(df):
    try:
        translator = GoogleTranslator(source='auto', target='en')

        def tr(x):
            try:
                return translator.translate(str(x)[:2000])
            except:
                return x

        df["Translated"] = df["Combined"].apply(tr)
        df["Translated"] = df["Translated"].apply(clean_text)
        return df, True
    except Exception as e:
        st.error(f"❌ Translation failed: {e}")
        return df, False

def cluster(df, threshold):
    try:
        model = load_embedding_model()
        texts = df["Combined"].fillna("").astype(str).tolist()

        emb = model.encode(texts, convert_to_numpy=True, batch_size=32)
        emb = normalize(emb)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold
        )

        df["Cluster"] = clustering.fit_predict(emb)
        return df, True
    except Exception as e:
        st.error(f"❌ Clustering failed: {e}")
        return df, False

def show_preview(df, title):
    st.success(f"✅ {title} completed successfully")
    st.dataframe(df.head(5))

def to_excel(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    buffer.seek(0)
    return buffer

st.title("📊 Data Cleaning & NLP Pipeline")

file = st.file_uploader("Upload Excel", type=["xlsx"])

if file:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox("Step 1 — Select Sheet", xls.sheet_names)

    df = pd.read_excel(file, sheet_name=sheet)
    st.session_state.data = df

    st.write("### Step 1 Preview (Raw Data)")
    st.dataframe(df.head(5))

    # =====================================================
    # STEP 2 — COMBINED
    # =====================================================
    st.header("Step 2 — Create Combined Column")

    cols = st.multiselect("Select columns", df.columns)

    skip_combined = st.button("Skip Step 2")

    if st.button("Run Step 2: Combine"):
        if len(cols) == 0:
            st.error("❌ Select at least 1 column")
        else:
            df, ok = create_combined_column(df, cols)
            if ok:
                st.session_state.data = df
                st.success("✅ Step 2 Completed: Combined column created")
                st.dataframe(df.head(5))

    if skip_combined:
        st.info("⏭ Step 2 skipped")

    # =====================================================
    # STEP 3 — DUPLICATES
    # =====================================================
    st.header("Step 3 — Remove Duplicates")

    exclude = st.multiselect("Exclude columns", df.columns)

    skip_dup = st.button("Skip Step 3")

    if st.button("Run Step 3: Remove Duplicates"):
        df, ok = remove_duplicates(df, exclude)
        if ok:
            st.session_state.data = df
            st.success("✅ Step 3 Completed: Duplicates removed")
            st.dataframe(df.head(5))

    if skip_dup:
        st.info("⏭ Step 3 skipped")

    # =====================================================
    # STEP 4 — KEYWORDS
    # =====================================================
    st.header("Step 4 — Keyword Matching")

    kw = st.file_uploader("Upload Keyword File", type=["xlsx"])

    output_col = st.text_input("Output Column Name", "Keyword Match")

    skip_kw = st.button("Skip Step 4")

    if st.button("Run Step 4: Keywords"):
        if kw is None:
            st.error("❌ Upload keyword file first")
        else:
            kw_df = pd.read_excel(kw)
            df, ok = keyword_match(df, kw_df, output_col)
            if ok:
                st.session_state.data = df
                st.success("✅ Step 4 Completed: Keyword matching done")
                st.dataframe(df.head(5))

    if skip_kw:
        st.info("⏭ Step 4 skipped")

    # =====================================================
    # STEP 5 — TRANSLATION
    # =====================================================
    st.header("Step 5 — Translation")

    skip_tr = st.button("Skip Step 5")

    if st.button("Run Step 5: Translate"):
        if "Combined" not in df.columns:
            st.error("❌ Combined column missing")
        else:
            df, ok = translate(df)
            if ok:
                st.session_state.data = df
                st.success("✅ Step 5 Completed: Translation done")
                st.dataframe(df.head(5))

    if skip_tr:
        st.info("⏭ Step 5 skipped")

    # =====================================================
    # STEP 6 — CLUSTERING
    # =====================================================
    st.header("Step 6 — Clustering")

    thr = st.slider("Cluster strictness", 0.1, 1.0, 0.28)

    skip_cluster = st.button("Skip Step 6")

    if st.button("Run Step 6: Cluster"):
        if "Combined" not in df.columns:
            st.error("❌ Combined column missing")
        else:
            df, ok = cluster(df, thr)
            if ok:
                st.session_state.data = df
                st.success("✅ Step 6 Completed: Clustering done")
                st.dataframe(df.head(5))

    if skip_cluster:
        st.info("⏭ Step 6 skipped")

    # =====================================================
    # FINAL OUTPUT
    # =====================================================
    st.header("Final Output")

    final_df = st.session_state.data

    st.write("### Final Preview")
    st.dataframe(final_df.head(5))

    out = to_excel(final_df)

    st.download_button(
        "📥 Download Excel",
        data=out,
        file_name="output.xlsx"
    )
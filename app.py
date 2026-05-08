# streamlit_app.py

```python
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
from openpyxl import load_workbook
from openpyxl.styles import Font

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(
    page_title="Media Cleaning & NLP Pipeline",
    layout="wide"
)

DetectorFactory.seed = 0

# ==========================================
# SESSION STATE INIT
# ==========================================
if "keyword_groups" not in st.session_state:
    st.session_state.keyword_groups = []

if "data" not in st.session_state:
    st.session_state.data = None

# ==========================================
# CACHED MODEL
# ==========================================
@st.cache_resource

def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    if pd.isnull(text):
        return text

    text = str(text)
    text = text.replace('_x000D_', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')

    return text.strip()



def detect_language(text):
    try:
        return detect(str(text))
    except:
        return "unknown"



def create_combined_column(df, selected_columns):
    df["Combined"] = (
        df[selected_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    df["Combined"] = df["Combined"].apply(clean_text)

    return df



def remove_duplicates_excel_style(df, excluded_columns):
    subset_cols = [c for c in df.columns if c not in excluded_columns]

    before = len(df)

    df = df.drop_duplicates(subset=subset_cols)

    after = len(df)

    return df, before - after



def extract_matching_sentences(text, keywords):
    if pd.isnull(text):
        return ""

    text = str(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)

    matches = []

    for sentence in sentences:
        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                matches.append(sentence.strip())
                break

    return "\n".join(matches)



def perform_keyword_matching(df, keyword_df, output_column):
    keyword_list = keyword_df.iloc[:, 0].dropna().astype(str).tolist()

    df[output_column] = df["Combined"].apply(
        lambda x: extract_matching_sentences(x, keyword_list)
    )

    return df



def translate_combined_column(df):
    languages = df['Combined'].apply(detect_language)
    language_counts = languages.value_counts(normalize=True)

    if language_counts.get('en', 0) + language_counts.get('ms', 0) >= 0.8:
        df['Translated'] = df['Combined']
        return df

    translator = GoogleTranslator(source='auto', target='en')

    def translate_limited(text):
        if pd.notnull(text):
            try:
                return translator.translate(str(text)[:2000])
            except:
                return text

        return text

    df['Translated'] = df['Combined'].apply(translate_limited)

    df['Translated'] = df['Translated'].apply(clean_text)

    return df



def perform_clustering(df, threshold=0.28):
    if "Combined" not in df.columns:
        raise ValueError("Combined column not found")

    texts = df["Combined"].fillna("").astype(str).tolist()

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    embeddings = normalize(embeddings)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    labels = clustering.fit_predict(embeddings)

    df["Story"] = labels

    cluster_desc = {}

    for cluster_id in np.unique(labels):
        sample_text = df.loc[
            df["Story"] == cluster_id,
            "Combined"
        ].iloc[0]

        cluster_desc[cluster_id] = sample_text[:120] + "..."

    df["Story Description"] = df["Story"].map(cluster_desc)

    return df



def to_excel_download(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Processed_Data')

        workbook = writer.book
        worksheet = writer.sheets['Processed_Data']

        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter

            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column].width = adjusted_width

    output.seek(0)

    return output

# ==========================================
# UI
# ==========================================
st.title("📊 Media Data Cleaning & NLP Pipeline")

st.markdown("This app performs data cleaning, keyword extraction, translation, and clustering using the **Combined** column only.")

# ==========================================
# STEP 1 — FILE UPLOAD
# ==========================================
with st.container():
    st.header("Step 1 — Upload Excel File")

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx"]
    )

# ==========================================
# LOAD EXCEL
# ==========================================
if uploaded_file:

    excel_file = pd.ExcelFile(uploaded_file)

    sheet_names = excel_file.sheet_names

    # ==========================================
    # STEP 2 — DATA TYPE
    # ==========================================
    st.header("Step 2 — Select Data Type")

    data_type = st.radio(
        "Choose Data Type",
        ["Mainstream Data", "Social Data"]
    )

    st.session_state["data_type"] = data_type

    # ==========================================
    # STEP 3 — SHEET SELECTION
    # ==========================================
    st.header("Step 3 — Select Sheet")

    if len(sheet_names) > 1:
        selected_sheet = st.selectbox(
            "Choose Sheet",
            sheet_names
        )
    else:
        selected_sheet = sheet_names[0]
        st.success(f"Auto-selected sheet: {selected_sheet}")

    data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    st.session_state.data = data.copy()

    st.subheader("Data Preview")
    st.dataframe(data.head())

    # ==========================================
    # STEP 4 — REMOVE DUPLICATES
    # ==========================================
    st.header("Step 4 — Remove Duplicates")

    excluded_columns = st.multiselect(
        "Select columns to EXCLUDE from duplicate detection",
        options=data.columns.tolist()
    )

    if st.button("Remove Duplicates"):

        data, removed_count = remove_duplicates_excel_style(
            data,
            excluded_columns
        )

        st.session_state.data = data

        st.success(f"Removed {removed_count} duplicate rows")

    # ==========================================
    # STEP 5 — COLUMN SELECTION
    # ==========================================
    st.header("Step 5 — Select Columns for Combined")

    selected_columns = st.multiselect(
        "Select columns to combine",
        options=data.columns.tolist()
    )

    # ==========================================
    # STEP 6 — CREATE COMBINED
    # ==========================================
    st.header("Step 6 — Create Combined Column")

    if st.button("Create Combined Column"):

        if len(selected_columns) == 0:
            st.error("Please select at least one column")

        else:
            data = create_combined_column(
                data,
                selected_columns
            )

            st.session_state.data = data

            st.success("Combined column created successfully")

            st.dataframe(data[["Combined"]].head())

    # ==========================================
    # STEP 7 — KEYWORD MATCHING
    # ==========================================
    st.header("Step 7 — Keyword Matching")

    keyword_file = st.file_uploader(
        "Upload Keyword File",
        type=["xlsx"],
        key="main_keyword"
    )

    output_column_name = st.text_input(
        "Output Column Name",
        value="Keyword Match"
    )

    if st.button("Run Keyword Matching"):

        if "Combined" not in data.columns:
            st.error("Please create Combined column first")

        elif keyword_file is None:
            st.error("Please upload keyword file")

        else:
            keyword_df = pd.read_excel(keyword_file)

            with st.spinner("Running keyword matching..."):
                data = perform_keyword_matching(
                    data,
                    keyword_df,
                    output_column_name
                )

            st.session_state.data = data

            st.success("Keyword matching completed")

    # ==========================================
    # STEP 8 — MULTIPLE KEYWORD GROUPS
    # ==========================================
    st.header("Step 8 — Additional Keyword Groups")

    add_group = st.checkbox("Add additional keyword groups")

    if add_group:

        number_of_groups = st.number_input(
            "Number of Additional Groups",
            min_value=1,
            max_value=20,
            value=1
        )

        for i in range(number_of_groups):
            st.subheader(f"Keyword Group {i+1}")

            group_file = st.file_uploader(
                f"Upload Keyword File {i+1}",
                type=["xlsx"],
                key=f"group_file_{i}"
            )

            group_output = st.text_input(
                f"Output Column Name {i+1}",
                value=f"Keyword_Group_{i+1}",
                key=f"group_output_{i}"
            )

            if st.button(f"Run Group {i+1}"):

                if group_file is not None:
                    group_df = pd.read_excel(group_file)

                    data = perform_keyword_matching(
                        data,
                        group_df,
                        group_output
                    )

                    st.session_state.data = data

                    st.success(f"Completed keyword group {i+1}")

    else:
        st.info("Skipped additional keyword groups")

    # ==========================================
    # STEP 10 — IRRELEVANT FILTERING
    # ==========================================
    st.header("Step 9 — Irrelevant Keyword Filtering")

    use_irrelevant = st.checkbox(
        "Filter irrelevant keywords"
    )

    if use_irrelevant:

        irrelevant_file = st.file_uploader(
            "Upload Irrelevant Keyword File",
            type=["xlsx"]
        )

        if st.button("Run Irrelevant Filtering"):

            if irrelevant_file is not None:
                irrelevant_df = pd.read_excel(irrelevant_file)

                data = perform_keyword_matching(
                    data,
                    irrelevant_df,
                    "Irrelevant Matches"
                )

                st.session_state.data = data

                st.success("Irrelevant keyword filtering completed")

    # ==========================================
    # STEP 11 — TRANSLATION
    # ==========================================
    st.header("Step 10 — Translation")

    use_translation = st.radio(
        "Translate Combined column?",
        ["No", "Yes"]
    )

    if use_translation == "Yes":

        if st.button("Run Translation"):

            with st.spinner("Translating content..."):
                data = translate_combined_column(data)

            st.session_state.data = data

            st.success("Translation completed")

    else:
        st.info("Translation skipped")

    # ==========================================
    # STEP 12 — CLUSTERING
    # ==========================================
    st.header("Step 11 — Clustering")

    run_clustering = st.checkbox("Enable clustering")

    if run_clustering:

        threshold = st.number_input(
            "Clustering Strictness",
            value=0.28,
            step=0.01
        )

        st.caption("Lower value = more clusters | Higher value = fewer clusters")

        if st.button("Run Clustering"):

            with st.spinner("Generating embeddings and clustering..."):
                data = perform_clustering(
                    data,
                    threshold
                )

            st.session_state.data = data

            st.success("Clustering completed")

            cluster_count = data['Story'].nunique()

            st.info(f"Total clusters found: {cluster_count}")

    # ==========================================
    # FINAL DATAFRAME
    # ==========================================
    final_df = st.session_state.data

    st.header("Processed Data Preview")

    st.dataframe(final_df.head())

    # ==========================================
    # STEP 13 — OUTPUT FILE NAME
    # ==========================================
    st.header("Step 12 — Export Output")

    output_file_name = st.text_input(
        "Output File Name",
        value="processed_media_data"
    )

    # ==========================================
    # STEP 14 — DOWNLOAD
    # ==========================================
    excel_data = to_excel_download(final_df)

    st.download_button(
        label="📥 Download Processed Excel",
        data=excel_data,
        file_name=f"{output_file_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
```

---

# requirements.txt

```txt
streamlit
pandas
numpy
openpyxl
langdetect
deep-translator
sentence-transformers
scikit-learn
torch
```

---

# Run Instructions

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

# Recommended Folder Structure

```txt
project_folder/
│
├── streamlit_app.py
├── requirements.txt
└── sample_data/
```

---

# Additional Recommended Improvements (Optional)

## Future Enhancements

### 1. Faster Embeddings

Use:

```python
all-MiniLM-L6-v2
```

for faster clustering on large datasets.

---

### 2. GPU Acceleration

If using local GPU:

```python
model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2",
    device="cuda"
)
```

---

### 3. Advanced Duplicate Detection

Can later add:

* Fuzzy matching
* Cosine similarity deduplication
* URL normalization
* Headline normalization

---

### 4. Better Hyperlink Preservation

Current version preserves exported Excel structure.
For true hyperlink preservation, use:

* openpyxl workbook cloning
* direct worksheet copying

---

### 5. Streamlit Deployment

Free deployment options:

* Streamlit Community Cloud
* Hugging Face Spaces
* Render
* Railway

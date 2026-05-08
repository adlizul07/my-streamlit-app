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
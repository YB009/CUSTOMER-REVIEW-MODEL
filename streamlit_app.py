# streamlit_app.py

import streamlit as st
import pandas as pd
from review_logic import analyze_reviews

st.set_page_config(
    page_title="Customer Review Model",
    layout="wide"
)

st.title("Customer Review Sentiment Analyzer")

st.markdown(
    "Analyze customer reviews via **text input** or **drag & drop CSV upload**."
)

# ---------------- Sidebar ----------------
st.sidebar.header("Input Options")

input_mode = st.sidebar.radio(
    "Choose input method:",
    ["Text Input", "Upload CSV"]
)

# ---------------- Text Input Mode ----------------
if input_mode == "Text Input":
    reviews_text = st.text_area(
        "Enter customer reviews (one per line):",
        height=220
    )

    if st.button("Analyze Reviews"):
        if not reviews_text.strip():
            st.warning("Please enter at least one review.")
        else:
            reviews = reviews_text.split("\n")
            results = analyze_reviews(reviews)

            df = pd.DataFrame(results)
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download Results (CSV)",
                df.to_csv(index=False),
                "sentiment_results.csv",
                "text/csv"
            )

# ---------------- File Upload Mode ----------------
else:
    uploaded_file = st.file_uploader(
        "Drag & drop a CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)

        if df_input.shape[1] == 0:
            st.error("CSV file must contain at least one column.")
        else:
            column = st.selectbox(
                "Select the column containing reviews:",
                df_input.columns
            )

            if st.button("Analyze File"):
                reviews = df_input[column].tolist()
                results = analyze_reviews(reviews)

                df_results = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(df_results, use_container_width=True)

                st.download_button(
                    "Download Results (CSV)",
                    df_results.to_csv(index=False),
                    "sentiment_results.csv",
                    "text/csv"
                )

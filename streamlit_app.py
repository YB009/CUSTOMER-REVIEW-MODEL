# streamlit_app.py

import streamlit as st
from review_logic import analyze_reviews

st.set_page_config(
    page_title="Customer Review Model",
    layout="centered"
)

st.title("Customer Review Sentiment Analyzer")

st.markdown(
    """
    Enter customer reviews below (one review per line).
    The app will analyze sentiment using **TextBlob**.
    """
)

reviews_input = st.text_area(
    "Customer Reviews",
    height=220,
    placeholder="Great service!\nThe product quality was poor.\nDelivery was on time."
)

if st.button("Analyze Reviews"):
    if not reviews_input.strip():
        st.warning("Please enter at least one review.")
    else:
        reviews = reviews_input.split("\n")
        results = analyze_reviews(reviews)

        if not results:
            st.warning("No valid reviews found.")
        else:
            st.subheader("Analysis Results")

            for idx, result in enumerate(results, start=1):
                st.markdown(f"### Review {idx}")
                st.write(result["review"])
                st.write(f"**Sentiment:** {result['sentiment_label']}")
                st.write(f"**Score:** {result['sentiment_score']}")
                st.divider()

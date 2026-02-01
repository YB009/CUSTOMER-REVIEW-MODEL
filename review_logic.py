# review_logic.py

from textblob import TextBlob
import nltk

# Ensure required NLP resources are available
# This is necessary on Streamlit Cloud
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def analyze_reviews(reviews):
    """
    Analyze sentiment of customer reviews.

    Parameters
    ----------
    reviews : list of str
        List of customer review texts.

    Returns
    -------
    list of dict
        Each dict contains:
        - review
        - sentiment_score
        - sentiment_label
    """

    results = []

    for review in reviews:
        review = review.strip()
        if not review:
            continue

        blob = TextBlob(review)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            label = "Positive"
        elif polarity < 0:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "review": review,
            "sentiment_score": round(polarity, 3),
            "sentiment_label": label
        })

    return results

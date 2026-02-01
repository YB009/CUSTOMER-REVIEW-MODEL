# review_logic.py

from textblob import TextBlob
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def analyze_reviews(reviews):
    results = []

    for review in reviews:
        review = str(review).strip()
        if not review:
            continue

        polarity = TextBlob(review).sentiment.polarity

        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        results.append({
            "review": review,
            "sentiment": sentiment,
            "score": round(polarity, 3)
        })

    return results

from flask import Blueprint, request, jsonify, render_template
from lda.gibbs_lda import GibbsLDA
from lda.utils import preprocess, batch_preprocess
import traceback
from collections import Counter
from textblob import TextBlob

bp = Blueprint('api', __name__)

# Global variable to store analyzed reviews by sentiment
_analyzed_reviews_by_sentiment = {}

# Define your categories and associated keywords
TOPIC_CATEGORIES = {
    "Electronics": {"electronics", "device", "gadget", "battery", "screen", "charger", "phone", "laptop", "camera", "instruction", "usb", "speaker", "headphone", "tv", "monitor", "tablet", "remote", "wireless", "bluetooth"},
    "Appearance": {"clothing", "shirt", "pants", "dress", "jeans", "jacket", "wear", "fashion", "size", "fit", "color", "flaw", "sweater", "t-shirt", "skirt", "shorts", "sock", "shoe", "sneaker", "boot", "style", "material"},
    "Home Goods": {"home", "furniture", "sofa", "table", "chair", "kitchen", "bed", "appliance", "goods", "quality", "lamp", "couch", "mattress", "blanket", "pillow", "sheet", "towel", "cookware", "utensil", "microwave", "oven", "fridge", "vacuum"},
    "Shipping": {"shipping", "delivery", "arrived", "late", "fast", "slow", "package", "order", "tracking", "courier", "dispatch", "ship", "receive", "delay", "box", "parcel"},
    "Customer Service": {"service", "support", "help", "customer", "staff", "response", "return", "refund", "exchange", "agent", "representative", "call", "email", "contact", "complaint", "resolve", "assistance"},
    "Price & Value": {"price", "cost", "cheap", "expensive", "value", "worth", "deal", "affordable", "discount", "sale", "offer", "bargain", "savings", "promotion", "budget", "overpriced"},
    "Food & Grocery": {"food", "taste", "flavor", "fresh", "grocery", "fruit", "vegetable", "meat", "snack", "drink", "beverage", "juice", "milk", "bread", "cereal", "cooking", "bake", "spice", "sweet", "salty", "delicious", "rotten", "expired"},
    "Health & Beauty": {"health", "beauty", "skin", "hair", "cream", "lotion", "shampoo", "conditioner", "soap", "makeup", "cosmetic", "vitamin", "supplement", "wellness", "hygiene", "fragrance", "perfume", "scent", "cleanser", "toothpaste", "toothbrush"},
    "Toys & Kids": {"toy", "kids", "children", "game", "play", "fun", "puzzle", "doll", "lego", "block", "car", "truck", "action", "figure", "board", "educational", "learning", "child", "infant", "baby", "crib", "stroller"},
    "Sports & Outdoors": {"sport", "outdoor", "fitness", "exercise", "gym", "run", "walk", "bike", "bicycle", "ball", "yoga", "mat", "tent", "camp", "hike", "swim", "golf", "tennis", "basketball", "football", "baseball", "fishing", "gear", "equipment"},
    "Automotive": {"car", "auto", "automotive", "vehicle", "engine", "tire", "wheel", "oil", "brake", "battery", "truck", "motor", "repair", "garage", "seat", "dashboard", "mirror", "wiper", "fuel", "gas", "transmission"},
    "Books & Media": {"book", "novel", "read", "story", "author", "magazine", "comic", "media", "movie", "film", "music", "song", "album", "cd", "dvd", "blu-ray", "podcast", "audiobook", "literature", "chapter", "page"},
    "Pets": {"pet", "dog", "cat", "animal", "puppy", "kitten", "fish", "bird", "hamster", "bowl", "leash", "collar", "treat", "food", "toy", "cage", "aquarium", "litter", "groom", "vet", "veterinarian"},
    "Office & Stationery": {"office", "stationery", "pen", "pencil", "notebook", "paper", "folder", "file", "desk", "chair", "printer", "ink", "stapler", "clip", "marker", "eraser", "calendar", "organizer", "supply", "envelope"},
    "Travel": {"travel", "trip", "flight", "hotel", "luggage", "bag", "passport", "ticket", "reservation", "tour", "journey", "vacation", "holiday", "cruise", "guide", "map", "itinerary", "airport", "check-in", "boarding"},
    # Add more as needed
}

# Review range labels and thresholds
REVIEW_RANGES = [
    ("Very Good", 0.6, 1.0),
    ("Good", 0.2, 0.6),
    ("Average", -0.2, 0.2),
    ("Bad", -1.0, -0.2)
]

def review_sentiment_score(text):
    return TextBlob(text).sentiment.polarity  # Returns a float in [-1, 1]

def get_review_range_label(score):
    for label, low, high in REVIEW_RANGES:
        if low < score <= high:
            return label
    return "Average"

def get_top_words(reviews, n=10):
    all_tokens = []
    for r in reviews:
        all_tokens.extend(preprocess(r))
    counter = Counter(all_tokens)
    return [w for w, _ in counter.most_common(n)]

def get_summary(reviews, max_sentences=4):
    if not reviews:
        return "No reviews in this range."
    return " ".join(reviews[:max_sentences])

def analyze_topics_with_lda(reviews, num_topics=10, coherence_threshold=0.3):
    """Analyze topics using enhanced LDA with coherence metrics"""
    # Preprocess reviews
    processed_reviews = batch_preprocess(reviews)
    
    # Initialize and train LDA model
    lda = GibbsLDA(
        num_topics=num_topics,
        coherence_threshold=coherence_threshold
    )
    lda.fit(processed_reviews)
    
    # Get topics and coherence scores
    topics = lda.get_topics()
    coherence_scores = lda.get_topic_coherence_scores()
    
    # Map topics to categories based on word overlap
    topic_categories = []
    for topic_words, scores in zip(topics, coherence_scores):
        # Find best matching category
        best_category = None
        max_overlap = 0
        for category, keywords in TOPIC_CATEGORIES.items():
            overlap = len(set(topic_words) & keywords)
            if overlap > max_overlap:
                max_overlap = overlap
                best_category = category
        
        topic_categories.append({
            'category': best_category or 'Other',
            'words': topic_words,
            'coherence_scores': scores
        })
    
    return topic_categories, lda.get_topic_distributions()

@bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@bp.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data or 'reviews' not in data:
            return jsonify({'error': 'No reviews provided'}), 400
        reviews = data['reviews']
        if not isinstance(reviews, list):
            return jsonify({'error': 'Reviews must be a list'}), 400
        if len(reviews) == 0:
            return jsonify({'error': 'Please provide at least one review'}), 400

        # Classify reviews by sentiment range
        range_reviews = {label: [] for label, _, _ in REVIEW_RANGES}
        review_scores = []
        for r in reviews:
            score = review_sentiment_score(r)
            label = get_review_range_label(score)
            range_reviews[label].append(r)
            review_scores.append((r, score, label))

        # Store range_reviews globally for access by other endpoints
        global _analyzed_reviews_by_sentiment
        _analyzed_reviews_by_sentiment = range_reviews

        # Analyze topics for each sentiment range
        topic_details = []
        for label, _, _ in REVIEW_RANGES:
            these_reviews = range_reviews[label]
            if these_reviews:
                # Use enhanced LDA for topic analysis
                topics, distributions = analyze_topics_with_lda(
                    these_reviews,
                    num_topics=min(5, len(these_reviews)),  # Adjust number of topics based on review count
                    coherence_threshold=0.3
                )
                
                topic_details.append({
                    'title': label,
                    'topics': topics,
                    'distributions': distributions,
                    'count': len(these_reviews)
                })
            else:
                topic_details.append({
                    'title': label,
                    'topics': [],
                    'distributions': [],
                    'count': 0
                })

        topic_stats = {
            'total_reviews': len(reviews),
            'num_topics': sum(len(t['topics']) for t in topic_details),
            'avg_words_per_topic': sum(
                sum(len(topic['words']) for topic in t['topics'])
                for t in topic_details
            ) / sum(len(t['topics']) for t in topic_details) if any(t['topics'] for t in topic_details) else 0,
            'topic_distributions': [t['count'] / len(reviews) if len(reviews) > 0 else 0 for t in topic_details]
        }

        return jsonify({
            'topics': topic_details,
            'topic_stats': topic_stats
        })
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@bp.route('/get_examples', methods=['POST'])
def get_examples():
    try:
        data = request.json
        if not data or 'category' not in data or 'sentiment' not in data or 'words' not in data:
            return jsonify({'error': 'Invalid request data'}), 400

        category = data['category']
        sentiment = data['sentiment']
        topic_words = data['words']

        # Get reviews for the requested sentiment range
        relevant_reviews = _analyzed_reviews_by_sentiment.get(sentiment, [])

        # Find example reviews that contain at least one of the topic words
        example_reviews = []
        # Limit the number of examples to return
        max_examples = 5

        for review_text in relevant_reviews:
            # Simple check: does the review contain any of the top words?
            if any(word.lower() in review_text.lower() for word in topic_words):
                example_reviews.append({
                    'text': review_text,
                    'sentiment': sentiment,
                    'date': 'N/A' # You might want to add date handling if available
                })
                if len(example_reviews) >= max_examples:
                    break

        return jsonify({'examples': example_reviews})

    except Exception as e:
        print(f"Error in get_examples endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})
from flask import Blueprint, request, jsonify, render_template
from lda.gibbs_lda import GibbsLDA
from lda.utils import preprocess
import traceback

bp = Blueprint('api', __name__)

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

        # Preprocess reviews
        docs = [preprocess(r) for r in reviews]
        
        # Initialize and fit LDA model
        lda = GibbsLDA(num_topics=5)
        lda.fit(docs)
        
        # Get topics and their distributions
        topics = lda.get_topics()
        topic_distributions = lda.get_topic_distributions()
        
        # Calculate topic statistics
        topic_stats = {
            'total_reviews': len(reviews),
            'num_topics': len(topics),
            'avg_words_per_topic': sum(len(topic) for topic in topics) / len(topics),
            'topic_distributions': topic_distributions
        }
        
        return jsonify({
            'topics': topics,
            'topic_stats': topic_stats
        })
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})
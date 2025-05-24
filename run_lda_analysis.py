from lda import GibbsLDA, optimize_num_topics
from lda.utils import preprocess, batch_preprocess
import json
import argparse

def load_documents(file_path):
    """Load documents from various file formats"""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict):
                return [str(item) for item in data.values()]
    else:
        raise ValueError("Unsupported file format. Use .txt or .json files.")

def main():
    parser = argparse.ArgumentParser(description='Run LDA analysis with coherence metrics')
    parser.add_argument('--input', required=True, help='Input file path (.txt or .json)')
    parser.add_argument('--num-topics', type=int, default=10, help='Number of topics')
    parser.add_argument('--coherence-threshold', type=float, default=0.3, help='Coherence threshold')
    parser.add_argument('--coherence-weight', type=float, default=0.3, help='Weight for coherence in optimization')
    parser.add_argument('--optimize', action='store_true', help='Run topic number optimization')
    parser.add_argument('--min-topics', type=int, default=5, help='Minimum number of topics for optimization')
    parser.add_argument('--max-topics', type=int, default=20, help='Maximum number of topics for optimization')
    
    args = parser.parse_args()
    
    # Load and preprocess documents
    print("Loading and preprocessing documents...")
    raw_documents = load_documents(args.input)
    documents = batch_preprocess(raw_documents)
    
    if args.optimize:
        print(f"Optimizing number of topics between {args.min_topics} and {args.max_topics}...")
        best_k, results = optimize_num_topics(
            documents=documents,
            topic_range=range(args.min_topics, args.max_topics + 1),
            coherence_weight=args.coherence_weight
        )
        
        print("\nOptimization Results:")
        print(f"Best number of topics: {best_k}")
        print("\nDetailed results:")
        for result in results:
            print(f"Topics: {result['num_topics']}")
            print(f"  Perplexity: {result['perplexity']:.2f}")
            print(f"  Average Coherence: {result['avg_coherence']:.2f}")
            print(f"  Combined Score: {result['combined_score']:.2f}")
            print()
        
        num_topics = best_k
    else:
        num_topics = args.num_topics
    
    # Train final model
    print(f"\nTraining LDA model with {num_topics} topics...")
    lda = GibbsLDA(
        num_topics=num_topics,
        coherence_threshold=args.coherence_threshold
    )
    lda.fit(documents)
    
    # Get and display results
    print("\nTopic Analysis Results:")
    topics = lda.get_topics()
    coherence_scores = lda.get_topic_coherence_scores()
    
    for i, (topic_words, scores) in enumerate(zip(topics, coherence_scores)):
        print(f"\nTopic {i + 1}:")
        print(f"  Words: {', '.join(topic_words)}")
        print(f"  C_v Score: {scores['cv_score']:.3f}")
        print(f"  C_p Score: {scores['cp_score']:.3f}")
        print(f"  Combined Score: {scores['combined_score']:.3f}")
    
    # Save results to file
    results = {
        'num_topics': num_topics,
        'topics': topics,
        'coherence_scores': coherence_scores,
        'topic_distributions': lda.get_topic_distributions()
    }
    
    output_file = f"lda_results_{num_topics}_topics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 
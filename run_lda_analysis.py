from lda import GibbsLDA, optimize_num_topics
from lda.utils import preprocess, batch_preprocess
import json
import argparse
import pandas as pd

def load_documents(file_path, max_documents=None):
    """Load documents from various file formats with optional limit"""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
            return documents[:max_documents] if max_documents else documents
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = [str(item) for item in data]
            elif isinstance(data, dict):
                documents = [str(item) for item in data.values()]
            return documents[:max_documents] if max_documents else documents
    elif file_path.endswith('.csv'):
        try:
            # Try to read CSV with different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    # Read CSV in chunks to handle large files
                    chunk_size = 1000
                    documents = []
                    for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=chunk_size):
                        # Try to find a column that might contain reviews
                        review_columns = [col for col in chunk.columns if any(keyword in col.lower() 
                            for keyword in ['review', 'text', 'comment', 'feedback', 'description'])]
                        
                        if review_columns:
                            # Use the first matching column
                            documents.extend(chunk[review_columns[0]].astype(str).tolist())
                        else:
                            # If no review-like column found, use the first column
                            documents.extend(chunk.iloc[:, 0].astype(str).tolist())
                        
                        if max_documents and len(documents) >= max_documents:
                            return documents[:max_documents]
                    
                    return documents
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not read CSV file with any supported encoding")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    else:
        raise ValueError("Unsupported file format. Use .txt, .json, or .csv files.")

def main():
    parser = argparse.ArgumentParser(description='Run LDA analysis with coherence metrics')
    parser.add_argument('--input', required=True, help='Input file path (.txt or .json)')
    parser.add_argument('--num-topics', type=int, default=10, help='Number of topics')
    parser.add_argument('--coherence-threshold', type=float, default=0.3, help='Coherence threshold')
    parser.add_argument('--coherence-weight', type=float, default=0.3, help='Weight for coherence in optimization')
    parser.add_argument('--optimize', action='store_true', help='Run topic number optimization')
    parser.add_argument('--min-topics', type=int, default=5, help='Minimum number of topics for optimization')
    parser.add_argument('--max-topics', type=int, default=20, help='Maximum number of topics for optimization')
    parser.add_argument('--max-documents', type=int, help='Maximum number of documents to process')
    
    args = parser.parse_args()
    
    # Load and preprocess documents
    print("Loading and preprocessing documents...")
    raw_documents = load_documents(args.input, args.max_documents)
    print(f"Loaded {len(raw_documents)} documents")
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
    print("\n" + "="*80)
    print("TOPIC ANALYSIS RESULTS".center(80))
    print("="*80)
    
    topics = lda.get_topics()
    coherence_scores = lda.get_topic_coherence_scores()
    topic_distributions = lda.get_topic_distributions()
    
    # Print topic headers
    print("\nTOPIC HEADERS:")
    print("-"*80)
    for i, topic_words in enumerate(topics):
        # Create a topic header from the top 3 words
        header = " ".join(topic_words[:3]).upper()
        print(f"Topic {i+1}: {header}")
    print("-"*80)
    
    # Print detailed topic information
    print("\nDETAILED TOPIC ANALYSIS:")
    print("-"*80)
    for i, (topic_words, scores, dist) in enumerate(zip(topics, coherence_scores, topic_distributions)):
        print(f"\nTopic {i + 1} (Distribution: {dist:.1%}):")
        print(f"  Top Words: {', '.join(topic_words)}")
        print(f"  Coherence Scores:")
        print(f"    - C_v Score: {scores['cv_score']:.3f}")
        print(f"    - C_p Score: {scores['cp_score']:.3f}")
        print(f"    - Combined Score: {scores['combined_score']:.3f}")
        print("-"*80)
    
    # Save results to file
    results = {
        'num_topics': num_topics,
        'topics': topics,
        'coherence_scores': coherence_scores,
        'topic_distributions': topic_distributions,
        'topic_headers': [f"Topic {i+1}: {' '.join(words[:3]).upper()}" 
                         for i, words in enumerate(topics)]
    }
    
    output_file = f"lda_results_{num_topics}_topics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 
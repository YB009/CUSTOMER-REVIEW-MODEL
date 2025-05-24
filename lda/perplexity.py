import numpy as np
from lda.gibbs_lda import GibbsLDA

def compute_perplexity(lda_model, documents):
    # Calculate perplexity for the given model and documents
    # This is a simplified version; for production, use log-likelihood
    N = sum(len(doc) for doc in documents)
    log_likelihood = 0
    for d, doc in enumerate(documents):
        for w in doc:
            prob = np.sum(
                (lda_model.n_kw[:, w] / lda_model.n_k) * (lda_model.n_dk[d] / np.sum(lda_model.n_dk[d]))
            )
            log_likelihood += np.log(prob)
    return np.exp(-log_likelihood / N)

def optimize_num_topics(documents, topic_range, coherence_weight=0.3, **lda_kwargs):
    """
    Optimize number of topics using both perplexity and coherence metrics
    
    Args:
        documents: List of preprocessed documents
        topic_range: Range of topic numbers to try
        coherence_weight: Weight for coherence score in combined metric (0-1)
        **lda_kwargs: Additional arguments for GibbsLDA
    """
    best_score = float('inf')
    best_k = None
    results = []
    
    for k in topic_range:
        # Train model
        lda = GibbsLDA(num_topics=k, **lda_kwargs)
        lda.fit(documents)
        
        # Calculate perplexity
        perp = compute_perplexity(lda, lda.docs)
        
        # Calculate average coherence score
        coherence_scores = lda.get_topic_coherence_scores()
        avg_coherence = np.mean([score['combined_score'] for score in coherence_scores])
        
        # Combine metrics (lower is better for perplexity, higher is better for coherence)
        combined_score = (1 - coherence_weight) * perp - coherence_weight * avg_coherence
        
        results.append({
            'num_topics': k,
            'perplexity': perp,
            'avg_coherence': avg_coherence,
            'combined_score': combined_score
        })
        
        if combined_score < best_score:
            best_score = combined_score
            best_k = k
    
    return best_k, results

def get_topic_distributions(self):
    """Calculate the distribution of topics across documents"""
    topic_counts = np.zeros(self.num_topics)
    for doc_topics in self.Z:
        for topic in doc_topics:
            topic_counts[topic] += 1
    return (topic_counts / topic_counts.sum()).tolist()
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

def optimize_num_topics(documents, topic_range, **lda_kwargs):
    best_perplexity = float('inf')
    best_k = None
    for k in topic_range:
        lda = GibbsLDA(num_topics=k, **lda_kwargs)
        lda.fit(documents)
        perp = compute_perplexity(lda, lda.docs)
        if perp < best_perplexity:
            best_perplexity = perp
            best_k = k
    return best_k, best_perplexity

def get_topic_distributions(self):
    """Calculate the distribution of topics across documents"""
    topic_counts = np.zeros(self.num_topics)
    for doc_topics in self.Z:
        for topic in doc_topics:
            topic_counts[topic] += 1
    return (topic_counts / topic_counts.sum()).tolist()
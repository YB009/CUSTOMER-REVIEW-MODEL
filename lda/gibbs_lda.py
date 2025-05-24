import numpy as np
from collections import defaultdict
from .coherence import TopicCoherence

class GibbsLDA:
    def __init__(self, num_topics, alpha=0.1, beta=0.01, iterations=1000, coherence_threshold=0.3):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.coherence_threshold = coherence_threshold
        self.coherence_metrics = None

    def fit(self, documents):
        # Preprocess and initialize
        self.vocab = list(set(word for doc in documents for word in doc))
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.docs = [[self.word2id[w] for w in doc] for doc in documents]
        self.D = len(self.docs)
        self.W = len(self.vocab)
        self.Z = [[np.random.randint(self.num_topics) for _ in doc] for doc in self.docs]
        self.n_dk = np.zeros((self.D, self.num_topics)) + self.alpha
        self.n_kw = np.zeros((self.num_topics, self.W)) + self.beta
        self.n_k = np.zeros(self.num_topics) + self.W * self.beta

        # Initialize coherence metrics
        self.coherence_metrics = TopicCoherence(documents, self.vocab, self.word2id)

        # Initialize counts
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                z = self.Z[d][i]
                self.n_dk[d, z] += 1
                self.n_kw[z, w] += 1
                self.n_k[z] += 1

        # Gibbs sampling with coherence-based topic selection
        for it in range(self.iterations):
            for d, doc in enumerate(self.docs):
                for i, w in enumerate(doc):
                    z = self.Z[d][i]
                    self.n_dk[d, z] -= 1
                    self.n_kw[z, w] -= 1
                    self.n_k[z] -= 1

                    # Calculate topic probabilities with coherence adjustment
                    p_z = (self.n_kw[:, w] / self.n_k) * self.n_dk[d]
                    
                    # Get current topic words for each topic
                    topic_words = self.get_topics(top_n=10)
                    
                    # Adjust probabilities based on coherence scores
                    coherence_scores = np.array([
                        self.coherence_metrics.get_combined_coherence(words)
                        for words in topic_words
                    ])
                    
                    # Apply coherence threshold
                    coherence_mask = coherence_scores > self.coherence_threshold
                    if np.any(coherence_mask):
                        p_z[~coherence_mask] *= 0.1  # Penalize low coherence topics
                    
                    p_z /= np.sum(p_z)
                    new_z = np.random.choice(self.num_topics, p=p_z)
                    self.Z[d][i] = new_z

                    self.n_dk[d, new_z] += 1
                    self.n_kw[new_z, w] += 1
                    self.n_k[new_z] += 1

# Collaborative Filtering: Words "vote" for topics based on co-occurrence patterns.  
# Result: Emergent topics that balance local (per-doc) and global (corpus-wide) structure.
# Tradeoff: Topics must explain both word frequencies (n_kw) and document mixtures (n_dk).  Overfit to global word trends (ignoring document context), or Overfit to document-level noise (ignoring word semantics).

    def get_topics(self, top_n=10):
        topics = []
        for k in range(self.num_topics):
            top_words = np.argsort(self.n_kw[k])[::-1][:top_n]
            topics.append([self.id2word[i] for i in top_words])
        return topics

    def get_topic_distributions(self):
        """Calculate the distribution of topics across documents"""
        topic_counts = np.zeros(self.num_topics)
        for doc_topics in self.Z:
            for topic in doc_topics:
                topic_counts[topic] += 1
        return (topic_counts / topic_counts.sum()).tolist()
        
    def get_topic_coherence_scores(self, top_n=10):
        """Get coherence scores for all topics"""
        topics = self.get_topics(top_n=top_n)
        coherence_scores = []
        for topic_words in topics:
            cv_score = self.coherence_metrics.compute_cv_coherence(topic_words, top_n)
            cp_score = self.coherence_metrics.compute_cp_coherence(topic_words, top_n)
            combined_score = self.coherence_metrics.get_combined_coherence(topic_words, top_n)
            coherence_scores.append({
                'cv_score': cv_score,
                'cp_score': cp_score,
                'combined_score': combined_score
            })
        return coherence_scores
import numpy as np
from collections import defaultdict
from .coherence import TopicCoherence

class GibbsLDA:
    def __init__(self, num_topics, alpha=0.1, beta=0.01, iterations=100, coherence_threshold=0.3, 
                 convergence_threshold=0.001, early_stopping_patience=5):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.coherence_threshold = coherence_threshold
        self.convergence_threshold = convergence_threshold
        self.early_stopping_patience = early_stopping_patience
        self.coherence_metrics = None

    def fit(self, documents):
        # Preprocess and initialize
        print("Initializing model...")
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

        # Initialize coherence metrics with caching
        print("Initializing coherence metrics...")
        self.coherence_metrics = TopicCoherence(documents, self.vocab, self.word2id)
        self.coherence_cache = {}

        # Initialize counts
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                z = self.Z[d][i]
                self.n_dk[d, z] += 1
                self.n_kw[z, w] += 1
                self.n_k[z] += 1

        # Gibbs sampling with early stopping
        print("Starting Gibbs sampling...")
        prev_perplexity = float('inf')
        patience_counter = 0
        best_topics = None
        best_coherence = float('-inf')

        for it in range(self.iterations):
            # Process documents in batches for better memory usage
            for d, doc in enumerate(self.docs):
                for i, w in enumerate(doc):
                    z = self.Z[d][i]
                    self.n_dk[d, z] -= 1
                    self.n_kw[z, w] -= 1
                    self.n_k[z] -= 1

                    # Calculate topic probabilities
                    p_z = (self.n_kw[:, w] / self.n_k) * self.n_dk[d]
                    
                    # Get current topic words (cached)
                    topic_words = self.get_topics(top_n=10)
                    topic_key = tuple(tuple(words) for words in topic_words)
                    
                    if topic_key not in self.coherence_cache:
                        coherence_scores = np.array([
                            self.coherence_metrics.get_combined_coherence(words)
                            for words in topic_words
                        ])
                        self.coherence_cache[topic_key] = coherence_scores
                    else:
                        coherence_scores = self.coherence_cache[topic_key]
                    
                    # Apply coherence threshold
                    coherence_mask = coherence_scores > self.coherence_threshold
                    if np.any(coherence_mask):
                        p_z[~coherence_mask] *= 0.1
                    
                    p_z /= np.sum(p_z)
                    new_z = np.random.choice(self.num_topics, p=p_z)
                    self.Z[d][i] = new_z

                    self.n_dk[d, new_z] += 1
                    self.n_kw[new_z, w] += 1
                    self.n_k[new_z] += 1

            # Calculate perplexity for early stopping
            if it % 10 == 0:  # Check every 10 iterations
                current_perplexity = self._calculate_perplexity()
                print(f"Iteration {it + 1}/{self.iterations}, Perplexity: {current_perplexity:.2f}")
                
                # Check for convergence
                if abs(prev_perplexity - current_perplexity) < self.convergence_threshold:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at iteration {it + 1}")
                        break
                else:
                    patience_counter = 0
                
                # Save best topics based on coherence
                current_coherence = np.mean([s['combined_score'] for s in self.get_topic_coherence_scores()])
                if current_coherence > best_coherence:
                    best_coherence = current_coherence
                    best_topics = self.get_topics()
                
                prev_perplexity = current_perplexity

        # Restore best topics if available
        if best_topics is not None:
            print("Restoring best topics based on coherence...")
            self._restore_best_topics(best_topics)

    def _calculate_perplexity(self):
        """Calculate perplexity for early stopping"""
        log_likelihood = 0
        N = sum(len(doc) for doc in self.docs)
        
        for d, doc in enumerate(self.docs):
            for w in doc:
                p_w = np.sum(self.n_dk[d] * self.n_kw[:, w] / self.n_k)
                log_likelihood += np.log(p_w)
        
        return np.exp(-log_likelihood / N)

    def _restore_best_topics(self, best_topics):
        """Restore the best topics based on coherence scores"""
        # Update topic-word distributions
        for k, words in enumerate(best_topics):
            word_ids = [self.word2id[w] for w in words if w in self.word2id]
            self.n_kw[k] = self.beta
            for w_id in word_ids:
                self.n_kw[k, w_id] += 1
            self.n_k[k] = np.sum(self.n_kw[k])

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

# Collaborative Filtering: Words "vote" for topics based on co-occurrence patterns.  
# Result: Emergent topics that balance local (per-doc) and global (corpus-wide) structure.
# Tradeoff: Topics must explain both word frequencies (n_kw) and document mixtures (n_dk).  Overfit to global word trends (ignoring document context), or Overfit to document-level noise (ignoring word semantics).
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

class TopicCoherence:
    def __init__(self, documents, vocab, word2id):
        self.documents = documents
        self.vocab = vocab
        self.word2id = word2id
        self.word_freq = self._compute_word_frequencies()
        self.co_occurrence = self._compute_co_occurrence()
        
    def _compute_word_frequencies(self):
        """Compute word frequencies in the corpus"""
        word_freq = defaultdict(int)
        for doc in self.documents:
            for word in doc:
                word_freq[word] += 1
        return word_freq
    
    def _compute_co_occurrence(self):
        """Compute word co-occurrence matrix within a sliding window"""
        window_size = 10
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for doc in self.documents:
            for i in range(len(doc)):
                for j in range(max(0, i - window_size), min(len(doc), i + window_size + 1)):
                    if i != j:
                        co_occurrence[doc[i]][doc[j]] += 1
        
        return co_occurrence
    
    def compute_cv_coherence(self, topic_words, top_n=10):
        """
        Compute C_v coherence score using cosine similarity of word vectors
        based on PMI scores
        """
        if len(topic_words) < 2:
            return 0.0
            
        # Get top N words for the topic
        top_words = topic_words[:top_n]
        
        # Create PMI matrix
        pmi_matrix = np.zeros((len(top_words), len(top_words)))
        for i, word1 in enumerate(top_words):
            for j, word2 in enumerate(top_words):
                if i != j:
                    pmi = self._compute_pmi(word1, word2)
                    pmi_matrix[i, j] = max(0, pmi)  # Only use positive PMI
        
        # Compute cosine similarity between word vectors
        similarities = cosine_similarity(pmi_matrix)
        
        # Average the similarities
        coherence = np.mean(similarities)
        return coherence
    
    def compute_cp_coherence(self, topic_words, top_n=10):
        """
        Compute C_p coherence score using pointwise mutual information
        """
        if len(topic_words) < 2:
            return 0.0
            
        # Get top N words for the topic
        top_words = topic_words[:top_n]
        
        # Calculate PMI scores for all word pairs
        pmi_scores = []
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                pmi = self._compute_pmi(top_words[i], top_words[j])
                pmi_scores.append(pmi)
        
        # Return average PMI score
        return np.mean(pmi_scores) if pmi_scores else 0.0
    
    def _compute_pmi(self, word1, word2):
        """Compute Pointwise Mutual Information between two words"""
        # Get word frequencies
        freq1 = self.word_freq[word1]
        freq2 = self.word_freq[word2]
        co_freq = self.co_occurrence[word1][word2]
        
        # Total number of word occurrences
        total_words = sum(self.word_freq.values())
        
        if freq1 == 0 or freq2 == 0 or co_freq == 0:
            return 0.0
            
        # Calculate PMI
        pmi = np.log((co_freq * total_words) / (freq1 * freq2))
        return pmi
    
    def get_combined_coherence(self, topic_words, top_n=10):
        """
        Get combined coherence score using both C_v and C_p metrics
        """
        cv_score = self.compute_cv_coherence(topic_words, top_n)
        cp_score = self.compute_cp_coherence(topic_words, top_n)
        
        # Combine scores (you can adjust weights based on your needs)
        combined_score = 0.6 * cv_score + 0.4 * cp_score
        return combined_score 
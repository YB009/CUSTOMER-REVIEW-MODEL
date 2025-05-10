import numpy as np
from collections import defaultdict

class GibbsLDA:
    def __init__(self, num_topics, alpha=0.1, beta=0.01, iterations=1000):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

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

        # Initialize counts
        for d, doc in enumerate(self.docs):
            for i, w in enumerate(doc):
                z = self.Z[d][i]
                self.n_dk[d, z] += 1
                self.n_kw[z, w] += 1
                self.n_k[z] += 1

        # Gibbs sampling
        for it in range(self.iterations):
            for d, doc in enumerate(self.docs):
                for i, w in enumerate(doc):
                    z = self.Z[d][i]
                    self.n_dk[d, z] -= 1
                    self.n_kw[z, w] -= 1
                    self.n_k[z] -= 1

                    p_z = (self.n_kw[:, w] / self.n_k) * self.n_dk[d]
                    p_z /= np.sum(p_z)
                    new_z = np.random.choice(self.num_topics, p=p_z)
                    self.Z[d][i] = new_z

                    self.n_dk[d, new_z] += 1
                    self.n_kw[new_z, w] += 1
                    self.n_k[new_z] += 1

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
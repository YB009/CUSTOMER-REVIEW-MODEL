import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess(text):
    """
    Advanced text preprocessing including:
    - Lowercase conversion
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Special character removal
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove special characters and numbers
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def batch_preprocess(texts):
    """
    Preprocess a list of texts
    """
    return [preprocess(text) for text in texts]

def get_word_frequencies(tokens):
    """
    Calculate word frequencies from tokens
    """
    from collections import Counter
    return Counter(tokens)
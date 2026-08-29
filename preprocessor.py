import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data resources safely
for resource in ['stopwords', 'punkt', 'wordnet']:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"Warning downloading NLTK resource {resource}: {e}")

class TextPreprocessor:
    """
    NLP Preprocessing pipeline for Twitter text.
    Handles URL removal, user mentions, hashtags, punctuation,
    tokenization, stopword removal, and lemmatization using NLTK.
    """
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"the", "a", "an", "is", "it", "in", "on", "and", "or", "to", "for", "of", "with"}
            
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase text
        text = text.lower()
        
        # 2. Remove URLs (http, https, www)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove Twitter @user mentions
        text = re.sub(r'@\w+', '', text)
        
        # 4. Clean hashtag symbols (keep the topic word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # 5. Remove non-alphanumeric characters (keep basic whitespace)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 6. Collapse multiple spaces into single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text):
        cleaned = self.clean_text(text)
        
        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()
            
        # Filter stopwords and short tokens
        filtered_tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatize
        if self.lemmatizer:
            try:
                lemmatized_tokens = [self.lemmatizer.lemmatize(t) for t in filtered_tokens]
            except Exception:
                lemmatized_tokens = filtered_tokens
        else:
            lemmatized_tokens = filtered_tokens
            
        return " ".join(lemmatized_tokens)

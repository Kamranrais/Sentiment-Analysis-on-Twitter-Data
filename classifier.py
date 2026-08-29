import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.download('vader_lexicon', quiet=True)
except Exception:
    pass

class TwitterSentimentClassifier:
    """
    Supervised Machine Learning Classifier for Twitter Sentiment Analysis
    (Positive, Negative, Neutral) using TF-IDF and Scikit-Learn classifiers.
    """
    def __init__(self, classifier_type="logistic_regression"):
        self.classifier_type = classifier_type.lower()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        if self.classifier_type in ["nb", "naive_bayes"]:
            self.model = MultinomialNB(alpha=1.0)
        elif self.classifier_type in ["lr", "logistic_regression"]:
            self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def fit(self, X_train, y_train):
        """Fits TF-IDF vectorizer and trains classifier model."""
        X_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_tfidf, y_train)
        return self

    def predict(self, X):
        """Predicts sentiment classes for input text series/list."""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict(X_tfidf)

    def predict_proba(self, X):
        """Predicts class probabilities."""
        X_tfidf = self.vectorizer.transform(X)
        return self.model.predict_proba(X_tfidf)

    def evaluate(self, X_test, y_test):
        """Evaluates model performance and prints classification report."""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "report": report
        }
        return metrics, y_pred

    def save(self, filepath):
        """Saves model and vectorizer to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "classifier_type": self.classifier_type
        }, filepath)
        print(f"Classifier saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads saved classifier model from file."""
        data = joblib.load(filepath)
        instance = cls(classifier_type=data["classifier_type"])
        instance.vectorizer = data["vectorizer"]
        instance.model = data["model"]
        print(f"Classifier loaded successfully from {filepath}")
        return instance

class VADERBaselineAnalyzer:
    """
    Rule-based Sentiment Analyzer using NLTK VADER.
    """
    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            self.sia = None

    def analyze(self, text):
        if not self.sia:
            return "neutral", 0.0
            
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return sentiment, compound

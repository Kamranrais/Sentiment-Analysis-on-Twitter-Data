import argparse
import os
import pandas as pd
from src.preprocessor import TextPreprocessor
from src.classifier import TwitterSentimentClassifier, VADERBaselineAnalyzer

def analyze_tweet(tweet_text, model_path="models/best_sentiment_model.joblib"):
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Training model first...")
        from train import main as train_main
        train_main()

    classifier = TwitterSentimentClassifier.load(model_path)
    preprocessor = TextPreprocessor()
    vader = VADERBaselineAnalyzer()

    clean_text = preprocessor.preprocess(tweet_text)
    prediction = classifier.predict([clean_text])[0]
    probabilities = classifier.predict_proba([clean_text])[0]
    class_labels = classifier.model.classes_

    vader_sentiment, vader_score = vader.analyze(tweet_text)

    prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}

    print("=" * 60)
    print("🐦 TWITTER SENTIMENT ANALYSIS RESULT")
    print("=" * 60)
    print(f"  • Input Tweet     : \"{tweet_text}\"")
    print(f"  • Cleaned Tokens  : \"{clean_text}\"")
    print(f"  • Predicted Class : {prediction.upper()}")
    print("  • Class Confidence Probabilities:")
    for label, prob in prob_dict.items():
        print(f"      - {label.capitalize():<10}: {prob * 100:.2f}%")
    print(f"  • NLTK VADER Compound Polarity Score: {vader_score:+.4f} ({vader_sentiment})")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Predict Sentiment for Input Tweets/Texts")
    parser.add_argument("--text", type=str, default="This new feature is absolutely incredible! Love the performance improvement! 🔥", help="Input tweet/text to classify")
    parser.add_argument("--model", type=str, default="models/best_sentiment_model.joblib", help="Path to trained model artifact")
    args = parser.parse_args()

    analyze_tweet(args.text, model_path=args.model)

if __name__ == "__main__":
    main()

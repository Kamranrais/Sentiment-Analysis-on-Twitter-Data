import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.dataset_generator import load_or_generate_dataset
from src.preprocessor import TextPreprocessor
from src.classifier import TwitterSentimentClassifier, VADERBaselineAnalyzer
from src.visualizer import plot_confusion_matrix, plot_sentiment_distribution

def main():
    parser = argparse.ArgumentParser(description="Train Twitter Sentiment Analysis Model")
    parser.add_argument("--data", type=str, default="data/tweets.csv", help="Path to input tweets CSV dataset")
    parser.add_argument("--test-split", type=float, default=0.2, help="Train/test split ratio (default: 0.2)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 STARTING TWITTER SENTIMENT ANALYSIS TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    df = load_or_generate_dataset(data_path=args.data)
    print(f"Total dataset records: {len(df)}")
    plot_sentiment_distribution(df, title="Dataset Sentiment Class Distribution", save_path="plots/sentiment_distribution.png")

    # 2. Text Preprocessing
    print("\n[1/3] Preprocessing tweets (cleaning, stopword removal, lemmatization)...")
    preprocessor = TextPreprocessor()
    df['clean_tweet'] = df['tweet'].apply(preprocessor.preprocess)

    X = df['clean_tweet']
    y = df['sentiment']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_split, random_state=42, stratify=y)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("-" * 60)

    # 4. Train Multinomial Naive Bayes Classifier
    print("\n[2/3] Training Multinomial Naive Bayes Classifier...")
    nb_classifier = TwitterSentimentClassifier(classifier_type="naive_bayes")
    nb_classifier.fit(X_train, y_train)
    nb_metrics, nb_preds = nb_classifier.evaluate(X_test, y_test)

    print(f"  • Naive Bayes Test Accuracy: {nb_metrics['accuracy'] * 100:.2f}%")

    # 5. Train Logistic Regression Classifier
    print("\n[3/3] Training Logistic Regression Classifier...")
    lr_classifier = TwitterSentimentClassifier(classifier_type="logistic_regression")
    lr_classifier.fit(X_train, y_train)
    lr_metrics, lr_preds = lr_classifier.evaluate(X_test, y_test)

    print(f"  • Logistic Regression Test Accuracy: {lr_metrics['accuracy'] * 100:.2f}%")

    # 6. Evaluate VADER Baseline on original raw text
    vader = VADERBaselineAnalyzer()
    raw_test_tweets = df.loc[X_test.index, 'tweet']
    vader_preds = [vader.analyze(t)[0] for t in raw_test_tweets]
    vader_acc = (vader_preds == y_test).mean()
    print(f"  • NLTK VADER Rule-Based Baseline Accuracy: {vader_acc * 100:.2f}%")

    print("-" * 60)

    # 7. Model Selection & Saving
    if lr_metrics["accuracy"] >= nb_metrics["accuracy"]:
        best_model = lr_classifier
        best_name = "Logistic Regression"
        best_metrics = lr_metrics
    else:
        best_model = nb_classifier
        best_name = "Multinomial Naive Bayes"
        best_metrics = nb_metrics

    print(f"🏆 Best Performing Model: {best_name} ({best_metrics['accuracy'] * 100:.2f}% Accuracy)")
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_sentiment_model.joblib"
    best_model.save(model_path)

    # Save Confusion Matrix Plot
    plot_confusion_matrix(
        best_metrics["confusion_matrix"],
        labels=['positive', 'neutral', 'negative'],
        title=f"Confusion Matrix ({best_name})",
        save_path="plots/confusion_matrix.png"
    )

    print("\n✅ Training complete! All plots and models saved.")

if __name__ == "__main__":
    main()

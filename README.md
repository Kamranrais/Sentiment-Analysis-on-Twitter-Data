# 🐦 Sentiment Analysis on Twitter Data

An end-to-end Natural Language Processing (NLP) tool designed to classify Twitter data into **positive**, **negative**, or **neutral** sentiment classes. Built using **NLTK**, **Scikit-Learn**, and **VADER Sentiment Analysis**.

---

## 🌟 Features

- **Text Cleaning & Preprocessing**:
  - URL and handle (@mention) stripping.
  - Hashtag symbol removal while preserving topic keywords.
  - Lowercasing and non-alphanumeric noise removal.
  - Tokenization using `NLTK`.
  - Stopword filtering and lemmatization via `WordNetLemmatizer`.
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with unigrams and bigrams.
- **Machine Learning Classifiers**:
  - **Logistic Regression** (High-performance linear classifier)
  - **Multinomial Naive Bayes** (Classic probabilistic NLP classifier)
- **Rule-based Baseline**: **NLTK VADER** Sentiment Intensity Analyzer for polarity scoring comparison.
- **Visualizations**: Auto-generated Confusion Matrix heatmaps and Sentiment Class Distribution charts.

---

## 📁 Repository Structure

```
twitter-sentiment-analysis/
├── data/                    # Dataset directory (contains tweets.csv)
├── src/
│   ├── __init__.py
│   ├── dataset_generator.py # Synthetic dataset loader/generator
│   ├── preprocessor.py      # NLTK text normalization & cleaning
│   ├── classifier.py        # Machine learning & VADER classifiers
│   └── visualizer.py        # Performance plots & charts
├── models/                  # Saved model artifacts (.joblib)
├── plots/                   # Saved visual metrics (.png)
├── train.py                 # Main model training & evaluation pipeline
├── predict.py               # Inference CLI tool for custom tweet text
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Kamranrais/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
pip install -r requirements.txt
```

### 2. Training the Model

Run the training pipeline to evaluate classifiers and generate plots:

```bash
python train.py
```

### 3. Predicting Tweet Sentiments

Test custom tweets or texts interactively:

```bash
python predict.py --text "Absolutely loving this incredible product update! 🚀 #awesome"
python predict.py --text "Worst customer service ever, completely disappointed with the delay. 😡"
```

---

## 📊 Model Evaluation Results

| Model | Model Type | Test Accuracy |
| :--- | :--- | :---: |
| **Logistic Regression** | Supervised ML + TF-IDF | **~99.0%** |
| **Multinomial Naive Bayes** | Supervised ML + TF-IDF | **~98.0%** |
| **NLTK VADER** | Rule-based Lexicon | **~75.0%** |

---

## 🤝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# Sentiment-Analysis-on-Twitter-Data

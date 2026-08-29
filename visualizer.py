import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_confusion_matrix(cm, labels=['positive', 'neutral', 'negative'], title="Confusion Matrix", save_path="plots/confusion_matrix.png"):
    """Plots and saves confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Sentiment", fontsize=12)
    plt.ylabel("True Sentiment", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to '{save_path}'")

def plot_sentiment_distribution(df, target_col='sentiment', title="Twitter Sentiment Distribution", save_path="plots/sentiment_distribution.png"):
    """Plots and saves sentiment class counts histogram."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    counts = df[target_col].value_counts()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette="Set2")
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sentiment Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Sentiment distribution plot saved to '{save_path}'")

import os
import pandas as pd
import numpy as np

def generate_twitter_dataset(num_samples=1500, output_path="data/tweets.csv"):
    """
    Generates a rich, realistic synthetic Twitter sentiment dataset containing
    positive, negative, and neutral tweets with typical Twitter syntax (hashtags, @mentions, emojis).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pos_templates = [
        "Absolutely loving the new update! Great job team! 🎉 #tech #awesome",
        "Just had the best coffee of my life. What a fantastic morning! ☕❤️",
        "Super excited for the upcoming product launch! Can't wait! 🚀",
        "I am so proud of @user for winning the award! Well deserved success!",
        "This tool is an absolute game changer. Saved me hours of work today! 🙌",
        "Happy Friday everyone! Hope you all have an amazing weekend!",
        "Incredible customer support from @support! They solved my issue in minutes.",
        "The weather today is stunning! Perfect day for a walk in the park ☀️",
        "Mastering Python and ML feels so rewarding! Never stop learning. 💻✨",
        "What a stunning performance by the team today! 🏆 Total victory!"
    ]
    
    neg_templates = [
        "Worst customer service ever from @company. Completely disappointed. 😡",
        "The new update completely broke the app. Very frustrated right now! 👎",
        "Spent 2 hours waiting in line and got nothing. Total waste of time.",
        "Terrible experience with the product. High price, poor quality.",
        "Why is the service down again? @provider please fix this immediately!",
        "Extremely dissatisfied with the latest policy changes. Unbelievable.",
        "My order was lost and support is ignoring my emails. Unacceptable!",
        "So tired of constant bugs and system crashes. Fix your code! ❌",
        "Traffic is a nightmare today. Stuck for over an hour already. 🚗💥",
        "Deeply disappointed by the results. Expecting way better performance."
    ]
    
    neu_templates = [
        "Here is the daily market summary for today: https://t.co/xyz123 #finance",
        "The conference will start at 10:00 AM UTC tomorrow. Join live via link.",
        "Reading a new article on NLP and sentiment analysis algorithms.",
        "Check out the latest documentation update on the official website.",
        "The package arrived today as scheduled by the delivery tracking.",
        "Flight 402 has been rescheduled to terminal 3.",
        "A python tutorial explaining scikit-learn and NLTK pipeline setup.",
        "Quarterly financial earnings report scheduled for release next Tuesday.",
        "Discussion on future trends in cloud computing and data science.",
        "Weather forecast for tomorrow indicates mild clouds and 22C."
    ]
    
    samples_per_cat = num_samples // 3
    
    pos_tweets = np.random.choice(pos_templates, samples_per_cat)
    neg_tweets = np.random.choice(neg_templates, samples_per_cat)
    neu_tweets = np.random.choice(neu_templates, samples_per_cat)
    
    df_pos = pd.DataFrame({'tweet': pos_tweets, 'sentiment': 'positive'})
    df_neg = pd.DataFrame({'tweet': neg_tweets, 'sentiment': 'negative'})
    df_neu = pd.DataFrame({'tweet': neu_tweets, 'sentiment': 'neutral'})
    
    df = pd.concat([df_pos, df_neg, df_neu], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sample tweets saved to '{output_path}'")
    return df

def load_or_generate_dataset(data_path="data/tweets.csv"):
    if os.path.exists(data_path):
        print(f"Loading existing dataset from '{data_path}'...")
        df = pd.read_csv(data_path)
    else:
        print(f"Dataset '{data_path}' not found. Generating sample dataset...")
        df = generate_twitter_dataset(num_samples=1500, output_path=data_path)
    return df

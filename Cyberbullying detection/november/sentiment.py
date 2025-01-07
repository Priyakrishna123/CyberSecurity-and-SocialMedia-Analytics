
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Explicitly define the Hugging Face model to avoid warnings
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Your remaining code follows...


# Load the CSV file containing tweets and labels
df = pd.read_csv('predicted_tweets.csv')

# Initialize the VADER sentiment analyzer and Hugging Face sentiment pipeline
analyzer = SentimentIntensityAnalyzer()
sentiment_model = pipeline("sentiment-analysis")

# Function to get TextBlob sentiment polarity
def get_textblob_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to get VADER sentiment score
def get_vader_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

# Function to get Hugging Face sentiment
def get_transformer_sentiment(text):
    return sentiment_model(text)[0]['label']

# Applying the sentiment analysis techniques
df['TextBlob_Sentiment'] = df['tweet'].apply(get_textblob_sentiment)
df['VADER_Sentiment'] = df['tweet'].apply(get_vader_sentiment)
df['Transformer_Sentiment'] = df['tweet'].apply(get_transformer_sentiment)

# Function to categorize sentiment based on VADER score
def categorize_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Categorize based on VADER sentiment score
df['Sentiment_Category'] = df['VADER_Sentiment'].apply(categorize_sentiment)

# Save the DataFrame to a new CSV file with labeled sentiment and sentiment categories
df.to_csv('tweets_with_sentiment.csv', index=False)

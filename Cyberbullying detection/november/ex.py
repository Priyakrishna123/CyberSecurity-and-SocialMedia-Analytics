import pandas as pd
import re

# Function to preprocess tweet text
def preprocess_tweet(tweet):
    # Ensure the tweet is a string and handle NaN or missing values
    tweet = str(tweet) if pd.notnull(tweet) else ""
    
    # Remove mentions (words starting with @)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Remove unwanted characters: dots, commas, double quotes, and the letter 'm'
    tweet = re.sub(r'[.,\'"“”‘’m]', '', tweet)
    
    # Remove empty double quotes and extra spaces
    tweet = tweet.replace('""', '').strip()
    
    # Remove extra spaces (if any)
    tweet = ' '.join(tweet.split())
    
    return tweet

# Read the CSV file into a DataFrame
file_path = 'elon musk_tweets.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Assuming the column containing tweets is named 'Tweet'
if 'tweet' in df.columns:
    # Preprocess the 'Tweet' column
    df['tweet'] = df['tweet'].apply(preprocess_tweet)

    # Save only the cleaned tweets to a new CSV file without printing
    df[['tweet']].to_csv('cleaned_tweets.csv', index=False)
else:
    print("No 'Tweet' column found in the CSV file.")

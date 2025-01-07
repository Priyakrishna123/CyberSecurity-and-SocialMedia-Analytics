import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Streamlit file uploader to upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Check if the CSV has a column named 'Tweet' (you may need to adjust this depending on the CSV structure)
    if 'tweet' in df.columns:
        tweets = df['tweet'].tolist()  # Extract tweets from the 'Tweet' column
        
        # Sentiment Analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiments = [analyzer.polarity_scores(tweet)['compound'] for tweet in tweets]

        # Topic Modeling
        vectorizer = CountVectorizer(max_df=0.95, min_df=1, stop_words='english')
        X = vectorizer.fit_transform(tweets)
        lda = LatentDirichletAllocation(n_components=2, random_state=0)
        topics = lda.fit_transform(X)

        # Prepare data for the dashboard
        tweet_data = pd.DataFrame({
            "Tweet": tweets,
            "Sentiment Score": sentiments,
            "Topic": topics.argmax(axis=1)  # Assuming max score topic per tweet
        })

        # Display the dashboard
        st.title("Twitter Content Analysis Dashboard")
        st.write("Analyze sentiment and topics in tweets.")

        # Display table of tweet data
        st.table(tweet_data)

        # Additional Visualization (e.g., sentiment distribution)
        st.bar_chart(tweet_data['Sentiment Score'])
    else:
        st.error("The uploaded CSV file must contain a 'Tweet' column.")
else:
    st.write("Please upload a CSV file to proceed.")

# sentiment_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

# Load the CSV file with topics and sentiment data
df = pd.read_csv('tweets_with_topics.csv')

# Set up Streamlit app layout
st.set_page_config(page_title="Tweet Sentiment & Topic Analysis", layout="wide")
st.title("Tweet Sentiment & Topic Analysis Dashboard")
st.write("Explore sentiment distribution, topic trends, and more insights from the tweets dataset.")

# Sidebar Filters
st.sidebar.header("Filter Options")
topic_filter = st.sidebar.multiselect("Select Topic(s)", options=df['Dominant_Topic'].unique())
sentiment_filter = st.sidebar.multiselect("Select Sentiment Category", options=df['Sentiment_Category'].unique())

# Select Sentiment Score Column
sentiment_column = st.sidebar.selectbox(
    "Select Sentiment Score Column",
    options=['TextBlob_Sentiment', 'VADER_Sentiment', 'Transformer_Sentiment']
)

# Filter data based on user selection
filtered_df = df
if topic_filter:
    filtered_df = filtered_df[filtered_df['Dominant_Topic'].isin(topic_filter)]
if sentiment_filter:
    filtered_df = filtered_df[filtered_df['Sentiment_Category'].isin(sentiment_filter)]

# Display Basic Metrics
st.header("Basic Metrics")
st.write("Overview of tweets based on selected filters:")
total_tweets = len(filtered_df)
sentiment_counts = filtered_df['Sentiment_Category'].value_counts()
topic_counts = filtered_df['Dominant_Topic'].value_counts()

st.metric("Total Tweets", total_tweets)
st.write("Sentiment Distribution:")
st.write(sentiment_counts)

# Sentiment Distribution Plot
st.subheader("Sentiment Distribution")
fig = plt.figure(figsize=(10, 5))
sns.countplot(data=filtered_df, x='Sentiment_Category', palette="Set2")
plt.title("Sentiment Distribution of Tweets")
st.pyplot(fig)

# Topic Distribution Plot
st.subheader("Topic Distribution")
fig2 = plt.figure(figsize=(10, 5))
sns.countplot(data=filtered_df, x='Dominant_Topic', palette="coolwarm")
plt.title("Topic Distribution of Tweets")
st.pyplot(fig2)

# Word Cloud for Selected Sentiment
st.subheader("Word Cloud")
selected_sentiment = st.selectbox("Select Sentiment for Word Cloud", df['Sentiment_Category'].unique())
sentiment_text = " ".join(filtered_df[filtered_df['Sentiment_Category'] == selected_sentiment]['tweet'])

if sentiment_text.strip():  # Check if there is text to create the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate(sentiment_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
else:
    st.write("No tweets available for the selected sentiment category to generate a word cloud.")

# Sentiment vs. Topic Scatter Plot
st.subheader("Sentiment vs. Topic Scatter Plot")
fig3 = px.scatter(filtered_df, x='Dominant_Topic', y=sentiment_column, color='Sentiment_Category',
                  title="Sentiment Scores by Topic",
                  labels={'Dominant_Topic': 'Topic', sentiment_column: 'Sentiment Score'},
                  color_discrete_sequence=px.colors.qualitative.Pastel)
st.plotly_chart(fig3)

# Display Data Table
st.subheader("Filtered Data")
st.write("Tweets data with sentiment and topic information based on selected filters:")
st.dataframe(filtered_df)





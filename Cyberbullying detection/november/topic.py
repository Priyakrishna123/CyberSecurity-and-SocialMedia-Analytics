import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV file with tweets and sentiment data
df = pd.read_csv('tweets_with_sentiment.csv')

# Preprocess text by tokenizing, converting to lowercase, and removing stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stop words and non-alphabetic tokens
    return tokens

# Apply preprocessing to each tweet
df['Tokens'] = df['tweet'].apply(preprocess_text)

# Create a dictionary and a corpus for the LDA model
dictionary = corpora.Dictionary(df['Tokens'])
corpus = [dictionary.doc2bow(text) for text in df['Tokens']]

# Set the number of topics for LDA (e.g., 5 topics)
num_topics = 5

# Train the LDA model on the corpus
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)

# Function to get the dominant topic for each tweet
def get_dominant_topic(bow):
    topic_scores = lda_model.get_document_topics(bow)
    dominant_topic = max(topic_scores, key=lambda x: x[1])[0]  # Select topic with the highest probability
    return dominant_topic

# Assign the dominant topic to each tweet
df['Dominant_Topic'] = df['Tokens'].apply(lambda x: get_dominant_topic(dictionary.doc2bow(x)))

# Save the output to a new CSV file with topics
df.to_csv('tweets_with_topics.csv', index=False)

print("Topic modeling complete. Output saved to 'tweets_with_topics.csv'.")

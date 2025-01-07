import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Function to clean text
def clean_text(text):
    text = str(text) if pd.notnull(text) else ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#\w+', '', text)     # Remove hashtags
    text = re.sub(r'\W', ' ', text)      # Remove non-alphanumeric characters
    return text.lower().strip()          # Convert to lowercase and strip whitespace

# Load the dataset
df = pd.read_csv('tweet_labels.csv')

# Clean the 'tweet' column
df['tweet'] = df['tweet'].apply(clean_text)

# Define labels
labels = [
    'Cyberbullying', 'Hate Speech', 'Offensive Language', 'Positive Sentiment',
    'Negative Sentiment', 'Neutral Sentiment', 'Personal Attacks',
    'Harassment', 'Abusive Language', 'Other'
]

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
df['label'] = label_encoder.transform(df['label'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

# Number of classes
num_classes = len(labels)

# Define the CNN + LSTM model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_seq, y_train, epochs=100, batch_size=32, validation_data=(X_test_seq, y_test))

# Load new data for predictions
new_data = pd.read_csv('cleaned_tweets.csv')
new_data['tweet'] = new_data['tweet'].apply(clean_text)

# Preprocess new data
new_data_seq = pad_sequences(tokenizer.texts_to_sequences(new_data['tweet']), maxlen=100)

# Make predictions
predictions = model.predict(new_data_seq)
predicted_labels = predictions.argmax(axis=1)

# Decode labels back to original category names
new_data['label'] = label_encoder.inverse_transform(predicted_labels)

# Save the results to a new CSV file
new_data.to_csv('predicted_tweets.csv', index=False)

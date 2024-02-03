# Feature Engineering 

# Sentimental Analysis Feature

# Why using it - Sentiment analysis can provide insights into the tone of the essays, which might help distinguish between student-written and LLM-generated texts.

import pandas as pd
import re  # Import regular expressions module
from textblob import TextBlob

# Define the path to the data file
train_essay_path = 'data/raw/train_essays.csv'

# Load the data
train_essays = pd.read_csv(train_essay_path)

# Define the text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply the cleaning function to create 'cleaned_text'
train_essays['cleaned_text'] = train_essays['text'].apply(clean_text)

# Define a function to calculate sentiment
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply the sentiment calculation function
train_essays['sentiment'] = train_essays['cleaned_text'].apply(calculate_sentiment)

# Print the sentiment calculation function 

# print(train_essays[['text', 'cleaned_text', 'sentiment']].head())

# import os

# current_directory = os.getcwd()
# print("Current directory:", current_directory)

# Additional Text-Based Features

# Text Length (Number of words)
# Why - The length of an essay could indicate complexity or depth of content.
# Alternatives: Character count or sentence count. However, word count is a good balance between detail and simplicity. 

train_essays['word_count'] = train_essays['cleaned_text'].apply(lambda x: len(x.split()))

# Lexical Diversity (Ratio of Unique Words to Total Words)
# Why: To measure the range of vocabulary used in the essay.
# Alternatives: Readability scores or syntactic complexity measures could also reflect language use but are more complex to compute

def calculate_laxical_diversity(text):
    words = text.split()
    return len(set(words)) / len (words) if words else 0

train_essays['lexical_diversity'] = train_essays['cleaned_text'].apply(calculate_laxical_diversity)

# Save the enhanced DataFrame to a new file 
enhanced_data_path = 'data/processed/enhanced_train_essays.csv'
train_essays.to_csv(enhanced_data_path, index=False)

# Print to verify
print(train_essays.head())
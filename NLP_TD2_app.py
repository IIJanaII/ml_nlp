import streamlit as st
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize

# Load NLTK resources
nltk.download('sentiwordnet')
nltk.download('punkt')

# Define functions to calculate sentiment scores
def get_adverbs(text):
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    adverbs = [word for word, tag in tags if tag.startswith('RB') or tag.startswith('JJ')]
    return adverbs

def calculate_sentiment_score(adverbs):
    sentiment_score = 0.0
    for adverb in adverbs:
        synsets = list(swn.senti_synsets(adverb))
        if synsets:
            sentiment_score += synsets[0].pos_score() - synsets[0].neg_score()
    return sentiment_score

# Streamlit UI
st.title("Janany's Movie Review Sentiment Analysis")

# User input
review = st.text_area("Enter your movie review:")

# Sentiment analysis
if st.button("Analyze"):
    if review:
        adverbs = get_adverbs(review)
        sentiment_score = calculate_sentiment_score(adverbs)
        
        if sentiment_score > 0:
            result = "Positive"
        else:
            result = "Negative"
        
        st.write(f"Sentiment: {result}")
    else:
        st.warning("Please enter a movie review.")

st.write("This app analyzes the sentiment of a movie review based on adjectives and adverbs.")

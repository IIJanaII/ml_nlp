import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
matplotlib.use("Agg")

# Set up the Streamlit app
st.title("NLP Toolkit App")

# Page 1: Verify Zipf's Law
if st.button("Go to Page 1"):
    st.header("Page 1: Verify Zipf's Law")
    
    # Text Input
    user_text = st.text_area("Enter text for Zipf's Law analysis")
    
    if user_text:
        # Tokenization and word frequency count
        words = nltk.word_tokenize(user_text.lower())
        word_freq = Counter(words)
        freqs = list(word_freq.values())
        
        # Zipf's Law Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(range(1, len(freqs) + 1), sorted(freqs, reverse=True, key=lambda x: -x))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Zipf's Law")
        ax.set_xlabel("Rank (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        st.pyplot(fig)



# Page 2: Text Preprocessing
if st.button("Go to Page 2"):
    st.header("Page 2: Text Preprocessing")
    
    # Text Input
    user_text = st.text_area("Enter text for preprocessing")
    
    # Preprocessing Options
    lower = st.checkbox("Lowercase")
    remove_stopwords = st.checkbox("Remove Stopwords")
    remove_punctuation = st.checkbox("Remove Punctuation")
    tokenize = st.checkbox("Tokenize")
    
    if st.button("Preprocess"):
        if user_text:
            # Preprocessing
            if lower:
                user_text = user_text.lower()
            
            if remove_punctuation:
                user_text = ''.join([char for char in user_text if char.isalnum() or char.isspace()])
            
            if remove_stopwords:
                nltk.download('stopwords')
                stop_words = set(stopwords.words('french'))  # Change to the appropriate language
                words = user_text.split()
                user_text = ' '.join([word for word in words if word not in stop_words])
            
            if tokenize:
                user_text = ' '.join(nltk.word_tokenize(user_text))
            
            st.write("Processed Text:")
            st.write(user_text)

# Page 3: Text Similarity
if st.button("Go to Page 3"):
    st.header("Page 3: Text Similarity")
    
    # Text Input for Two Texts
    text1 = st.text_area("Enter Text 1")
    text2 = st.text_area("Enter Text 2")
    
    # Similarity Measure Options
    similarity_measure = st.selectbox("Choose a similarity measure", ("Cosine Similarity", "Euclidean Distance"))
    
    if st.button("Calculate Similarity"):
        if text1 and text2:
            if similarity_measure == "Cosine Similarity":
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
            else:
                words1 = nltk.word_tokenize(text1)
                words2 = nltk.word_tokenize(text2)
                vector1 = np.array([len(words1), len(words2)]).reshape(1, -1)
                similarity = euclidean_distances(vector1)

            st.write("Similarity Measure:", similarity[0][0])


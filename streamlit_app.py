"""
Streamlit application for spam email classification.
Data source: https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

# Sample emails for quick testing
SAMPLE_EMAILS = {
    "None": "",
    "Normal Email 1": "Dear valued customer, Your account statement is now available. Please log in to your secure account to view it.",
    "Normal Email 2": "Team meeting scheduled for tomorrow at 10 AM. Please prepare your weekly progress report.",
    "Spam Email 1": "CONGRATULATIONS! You've won $1,000,000 in our lottery! Click here to claim your prize now!!!",
    "Spam Email 2": "Buy now! 90% OFF on luxury watches! Limited time offer! Don't miss out!!!",
}

# Load training data
@st.cache_data
def load_training_data():
    try:
        # Read the SMS spam dataset
        df = pd.read_csv('sms_spam.csv', names=['label', 'text'])
        # Convert ham/spam to 0/1
        df['label'] = (df['label'] == 'spam').astype(int)
        return df['text'].tolist(), df['label'].tolist()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        # Fallback to minimal dataset if file loading fails
        emails = [
            "Your account statement is ready",
            "Meeting tomorrow at 10 AM",
            "Win million dollars now!!!",
            "90% OFF luxury watches!!!"
        ]
        labels = [0, 0, 1, 1]
        return emails, labels

def train_model(emails, labels):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(emails)
    model = MultinomialNB()
    model.fit(X, labels)
    return vectorizer, model

def plot_probability_bar(probability):
    fig, ax = plt.subplots(figsize=(10, 2))
    colors = ['green' if probability < 0.5 else 'red']
    sns.barplot(x=[probability], y=['Probability'], ax=ax, palette=colors)
    ax.set_xlim(0, 1)
    plt.title('Spam Classification Probability')
    st.pyplot(fig)
    plt.close()

def main():
    st.title('Email Spam Classification System')
    
    # Load and train model
    emails, labels = load_training_data()
    vectorizer, model = train_model(emails, labels)
    
    # Create a layout with two columns
    col1, col2 = st.columns([1, 3])
    
    # Sample selection in first column
    with col1:
        email_template = st.selectbox(
            "Select template",
            list(SAMPLE_EMAILS.keys())
        )
    
    # Text input in second column
    with col2:
        text_to_analyze = st.text_area(
            "Email content to analyze",
            value=SAMPLE_EMAILS[email_template],
            height=100
        )
    
    if st.button('Analyze') and text_to_analyze:
        # Feature extraction and prediction
        X_test = vectorizer.transform([text_to_analyze])
        spam_prob = model.predict_proba(X_test)[0][1]
        
        # Display results
        st.header('Analysis Results')
        result = "Spam" if spam_prob > 0.5 else "Ham"
        st.write(f"This email is likely: **{result}**")
        
        # Show probability bar
        plot_probability_bar(spam_prob)
        
        # Show detailed probabilities
        st.write(f"- Ham probability: {(1-spam_prob)*100:.2f}%")
        st.write(f"- Spam probability: {spam_prob*100:.2f}%")
        
        # Get words from current text
        current_text_features = vectorizer.transform([text_to_analyze])
        current_words = set()
        for idx, val in enumerate(current_text_features.toarray()[0]):
            if val > 0:
                current_words.add(vectorizer.get_feature_names_out()[idx])
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'word': vectorizer.get_feature_names_out(),
            'importance': model.feature_log_prob_[1] - model.feature_log_prob_[0]
        })
        
        # Only keep words that appear in current text
        feature_importance = feature_importance[feature_importance['word'].isin(current_words)]
        top_features = feature_importance.nlargest(5, 'importance')
        
        st.subheader('Key words influencing the decision')
        if not top_features.empty:
            for _, row in top_features.iterrows():
                st.write(f"- {row['word']}: {row['importance']:.4f}")
        else:
            st.write("No significant keywords found in the text")

if __name__ == '__main__':
    main()
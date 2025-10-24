"""
Streamlit application for spam email classification.
"""
import streamlit as st
import pandas as pd
import numpy as np
from src.data_processing.preprocessor import EmailPreprocessor
from src.models.classifier import SpamClassifier
from src.visualization.visualizer import SpamVisualizer

def main():
    st.title('Spam Email Classification')
    
    # Initialize components
    preprocessor = EmailPreprocessor()
    classifier = SpamClassifier()
    visualizer = SpamVisualizer()
    
    # Sidebar
    st.sidebar.header('Options')
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload training data (CSV)",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file)
        
        # Training section
        st.header('Model Training')
        if st.button('Train Model'):
            with st.spinner('Training in progress...'):
                # Preprocess
                X = preprocessor.fit_transform(df['text'].values)
                y = df['label'].values
                
                # Train
                classifier.fit(X, y)
                
                # Evaluate
                metrics = classifier.evaluate(X, y)
                
                # Display results
                st.success('Training completed!')
                visualizer.plot_metrics(metrics)
                visualizer.plot_confusion_matrix(y, classifier.predict(X))
    
    # Prediction section
    st.header('Email Classification')
    email_text = st.text_area('Enter email text to classify:')
    
    if email_text and st.button('Classify'):
        # Preprocess and predict
        X = preprocessor.transform([email_text])
        prediction = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]
        
        # Display result
        result = 'SPAM' if prediction == 1 else 'HAM'
        st.write(f'Classification: **{result}**')
        st.write(f'Confidence: {probabilities[prediction]:.2%}')
        
        # Display probability distribution
        visualizer.plot_probability_distribution(
            classifier.predict_proba(X)
        )

if __name__ == '__main__':
    main()
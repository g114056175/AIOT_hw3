# AIOT_hw3 - Spam Email Classification

This project implements a spam email classification system using machine learning, deployed on Streamlit Cloud.

## Live Demo
Visit the live application at: [Streamlit Cloud Demo](https://aiothw3-jycxbyewety9g3fogipmeq.streamlit.app/)

## Features
- Email text classification (Spam/Ham)
- Interactive web interface
- Real-time prediction
- Performance visualization
- Model metrics display

## Project Structure
```
.
├── src/
│   ├── data_processing/  # Data preprocessing modules
│   ├── models/          # ML model implementations
│   └── visualization/   # Visualization utilities
├── streamlit_app.py    # Main Streamlit application
└── requirements.txt    # Project dependencies
```

## How to Use
1. Visit the [live demo](https://aiothw3-jycxbyewety9g3fogipmeq.streamlit.app/)
2. Choose between the "Spam Detection Demo" and "Model Performance" pages
3. In the demo page:
   - Select a sample email from the dropdown or enter your own text
   - Click "Analyze" to see the classification results
   - View probability scores and influential keywords
4. In the performance page:
   - Check model accuracy metrics
   - Examine the confusion matrix

## Technologies Used
- Python
- Streamlit
- scikit-learn
- NLTK
- pandas
- matplotlib
- seaborn

## Reference
- Based on patterns and datasets from [Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git)
- Tutorial: [YouTube Link](https://www.youtube.com/watch?v=ANjiJQQIBo0)
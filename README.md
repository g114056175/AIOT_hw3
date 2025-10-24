# AIOT_hw3 - Spam Email Classification

This project implements a spam email classification system using machine learning, deployed on Streamlit Cloud.

## Live Demo
Visit the live application at: [Streamlit Cloud Demo](https://g114056175-aiot-hw3.streamlit.app)

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
1. Visit the [live demo](https://g114056175-aiot-hw3.streamlit.app)
2. Upload training data (CSV format with 'text' and 'label' columns)
3. Train the model using the provided interface
4. Enter email text to classify
5. View classification results and visualizations

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
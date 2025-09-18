# Real vs Fake Data Classifier + Chatbot

A Streamlit web application that combines machine learning classification with an intelligent chatbot.

## Features

### 1. Real vs Fake Data Classifier
- Generates synthetic datasets with different patterns (blobs, moons, multivariate normal)
- Trains multiple ML models (Logistic Regression, SVM, Random Forest, XGBoost)
- Provides comprehensive visualizations and performance metrics
- Supports both 2D (visualizable) and high-dimensional data

### 2. Intelligent Chatbot
- Uses OpenAI's GPT models for natural language processing
- Detects intent (factual vs creative queries)
- Maintains conversation memory
- Works in both API mode and local fallback mode

## Installation

1. Clone or download this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
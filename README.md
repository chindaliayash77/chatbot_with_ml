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
   git clone https://github.com/chindaliayash77/chatbot_with_ml.git
   pip install -r requirements.txt
   streamlit run app.py

## Project images
1. I have added the ui images in project_images/ of machine learning and chatbot working .

## chatbot.py
Intent Detection: The system first classifies a user's input as either a factual query or a creative prompt. It uses the LLM itself for this classification if an API key is provided; otherwise, it falls back to a simple keyword-based detection.

Conditional Responses: Based on the detected intent, the system directs the query to the appropriate handler function. This is the core of its "agentic" behavior, as it makes a decision about how to respond before generating text.

Factual Queries: Handled to provide accurate, informative answers.

Creative Prompts: Handled with a higher temperature setting to encourage more imaginative and less constrained responses.

Fallback Mechanism: The system is designed to work even without an OpenAI API key. In this limited mode, it provides pre-defined, hard-coded responses for a small set of queries and prompts. This ensures the system doesn't crash but operates with reduced functionality.

Short-Term Memory: The system maintains a brief conversation history to provide context for subsequent interactions. It keeps the last three user-assistant pairs in memory, allowing for contextual follow-up questions without being overwhelmed by a long conversation.

Special Commands: The system recognizes specific user commands like clear memory or history to manage the conversation state, adding a layer of control for the user.

## classifier.py

Core Purpose
The class is designed to build and evaluate machine learning models that can differentiate between real and synthetic data.

It handles two types of scenarios: a 2D case for easy visualization and a high-dimensional case for a more realistic problem.

Pipeline Stages
Data Generation:

generate_real_data: Creates complex, non-linear data patterns (e.g., blobs, moons) to simulate real-world data distributions.

generate_fake_data: Creates simpler, often random data patterns (e.g., uniform, noise) to represent synthetic data.

Dataset Preparation:

prepare_dataset: Combines the real and fake data and assigns them binary labels (1 for real, 0 for fake). It then splits this combined dataset into training and testing sets.

Model Training:

train_models: Trains four common machine learning models: Logistic Regression, SVM, Random Forest, and XGBoost.

It uses Scikit-learn Pipelines to automatically scale the data (StandardScaler) before training the models, which is a best practice.

Evaluation:

evaluate_models: Assesses the performance of each trained model on the test data.

It calculates key metrics like Accuracy, ROC AUC, and performs cross-validation to get a more reliable performance estimate.

Visualization and Summary:

get_plots: Generates various plots to visualize the data and model performance.

2D Plots: Shows data distribution and decision boundaries for low-dimensional data.

High-Dimensional Plots: Uses PCA to visualize the data in 2D and plots feature importance to explain the model's decisions.

Performance Plots: Creates charts comparing model accuracy and ROC AUC, as well as confusion matrices and ROC curves for a deeper look at model performance.

get_summary: Generates a formatted summary report with model rankings and a detailed classification report for the best model.

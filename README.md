# MLOps-in-Travel-Production-of-ML-Systems

# ✈️ Integrating MLOps in Travel Productionization of ML Systems

# 📌 Project Overview

This project demonstrates an end-to-end MLOps pipeline for automating the travel planning process using Machine Learning. The solution integrates flight price prediction, hotel recommendation, tourist attraction discovery, and user classification into a production-ready system. It combines machine learning model development with deployment, experiment tracking, REST APIs, containerization, and an interactive web application.

The project follows the complete machine learning lifecycle, including data preprocessing, feature engineering, model training, evaluation, deployment, model versioning, and user interaction through a Streamlit application.

# 🎯 Project Objectives
Predict flight prices using historical travel data.
Recommend hotels based on user preferences and travel history.
Discover tourist attractions for travel destinations.
Classify user gender using machine learning techniques.
Develop production-ready REST APIs using Flask.
Track experiments and model versions using MLflow.
Containerize the application using Docker.
Build an interactive Streamlit dashboard for visualization and recommendations.
Demonstrate an end-to-end MLOps workflow for travel analytics.

# ⚙️ Data Preprocessing

The following preprocessing techniques were applied:

Missing value handling
Duplicate removal
Date conversion
Feature encoding
Numerical feature normalization
Feature scaling
Feature engineering
Dataset merging
Data validation
# 🔍 Feature Engineering
Flight Features
Travel Month
Travel Day
Travel Weekday
Weekend Indicator
Flight Distance
Travel Duration
Flight Type
Airline Agency
Hotel Features
Average Hotel Cost
Price Per Day
Popularity Score
Travel Frequency
Previous Hotels
Destination
User Features
Age Group
Gender
Company
Travel Count
# 🤖 Machine Learning Models
Flight Price Prediction (Regression)

Implemented models:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
CatBoost Regressor
LightGBM Regressor
Evaluation Metrics
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
Mean Absolute Percentage Error (MAPE)
Gender Classification

Implemented models:

Logistic Regression
Random Forest Classifier
XGBoost Classifier
LightGBM Classifier
Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Hotel Recommendation System

Recommendation techniques:

Content-Based Filtering
Collaborative Filtering
Hybrid Recommendation System
Popularity-Based Recommendation

Recommendation features:

Place
Hotel Name
Price
Days
User Travel History
Popularity Score

# 📈 MLflow

MLflow is used for:

Experiment Tracking
Parameter Logging
Metric Logging
Artifact Management
Model Versioning
Model Registry

Tracked models include:

Random Forest
XGBoost
LightGBM
CatBoost

# 🌐 REST APIs
Flight Price Prediction API

# 📊 Streamlit Dashboard

The Streamlit application provides:

Flight Price Prediction
Hotel Recommendation
Travel Analytics


This project demonstrates how MLOps can be applied to build a scalable and production-ready travel intelligence platform. It integrates machine learning models for flight price prediction, gender classification, and hotel recommendation with Flask REST APIs, MLflow experiment tracking, Docker-based containerization, and an interactive Streamlit dashboard. The solution showcases the complete machine learning lifecycle from data preprocessing to deployment, enabling reliable, reproducible, and scalable travel analytics for real-world applications.

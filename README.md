# Recommendation_System

## Introduction

Recommendation Systems play a key role in the growth of an e-commerce company. E-Commerce giants like Amazon,flipkart have been using recommendation algorithms to improve their user experience. The project focuses on building a sentiment based recommendation system which will recommend upto 5 products for an existing user. The dataset used here is inspired from https://www.kaggle.com/datasets/datafiniti/grammar-and-online-product-reviews. It consists of a subset of the original dataset and has 30000 reviewes for 200+products. 

## Topics
1. Data Preprocessing and EDA
  - the data pre-processing and EDA is covered in notebooks/DataPreprocessing.ipynb
2. Experimentation of different Ml models for sentiment analysis
   - Experimentation of sentiment prediction for Bag of words and TF-IDF representation using the below models.
   * Naive Bayes
   * Logistic Regression
   * XGboost
   The model experimentation is covered in notebooks/Sentiment_prediction.ipynb
3. Exploring collaborative filtering algorithms for recommendation
   * User-User based collaborative filtering
   * Item -Item based Collaborative filtering
   Recommendation models are covered in notebooks/Recommendation_System.ipynb
4. Model deployment using flask framework

## Project setup

### Installing the packages

For running the project, please install the necessary packages mentioned in the requirements.txt using the below command

'pip install -r requirements.txt'

### Run the main application
To run the flask app please run app.py file.

### Improvements 

* The dataset had a huge class imbalance. This can be handled by adding synthetic data for negative reviews.
* In the data pre-processing notebooks only BOW and TF_IDF representation was experimented. Word embedding models can be explored , as it focuses on the understanding the similarity between words.
* The project only recommends products for a user trained in the dataset. Better algorithms and approaches can be used to build a more realistic recommendation system. 

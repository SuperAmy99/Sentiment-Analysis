# Exploring Emotions in Tweets: A Sentiment Analysis Project

## Introduction

Sentiment Analysis is a technique that leverages data analysis to uncover the emotional tone behind text data. In this project, I explore the world of sentiment analysis and its real-world applications.

## Project Overview

This is an end-to-end machine learning project that covers the entire process of Twitter sentiment analysis, including data collection, text cleaning, modeling, data visualization, and presentation.

### Data Collection

For data collection, I utilized two sources:

1. **Kaggle Dataset**: I used a public dataset from Kaggle to gather historical tweets for sentiment analysis. You can access the dataset using the following link:

   [Kaggle Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/data)

2. **Scraper API**: I collected real-time tweets using the [Scraper API](https://dashboard.scraperapi.com/), which allowed me to access and gather tweets from Twitter.

### Data Modeling

#### Text Preprocessing

- Text cleaning involved removing noise and irrelevant characters from the text data.
- Tokenization: I split text into words or tokens to prepare it for further analysis.
- Stemming and Lemmatization: I applied stemming and lemmatization techniques to reduce words to their base or root form.
- Stopword Removal: I removed common stopwords from the text data to focus on meaningful words.

#### Machine Learning Models

I tested several machine learning models for sentiment analysis, including:

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)

#### Hyperparameter Tuning

To achieve optimal results, I fine-tuned the hyperparameters of the selected model (Logistic Regression). This process involved experimenting with different parameter settings to improve accuracy and performance.

### Data Modeling Notebook

You can explore the data modeling process in this [Jupyter Notebook](https://github.com/SuperAmy99/Sentiment-Analysis/blob/main/Sentiment%20Analysis%20-%20Machine%20Learning.ipynb).

### Data Visualization

I visualized the data using Python and Streamlit:

1. Python Data Visualization Notebook: You can explore the data visualization process in this [Jupyter Notebook](https://github.com/SuperAmy99/Sentiment-Analysis/blob/main/Sentiment%20Analysis%20-%20Data%20Visualization.ipynb).

2. Streamlit Application: I developed a Streamlit application to present the sentiment analysis results. You can find the source code for the Streamlit app in [app.py](https://github.com/SuperAmy99/Sentiment-Analysis/blob/main/app.py) and the helper functions in [helper_functions.py](https://github.com/SuperAmy99/Sentiment-Analysis/blob/main/helper_functions.py).

### Presentation

For a comprehensive presentation of this project, please refer to my [Twitter Sentiment Analysis PDF](https://github.com/SuperAmy99/Sentiment-Analysis/blob/main/twitter%20sentiment%20analysis.pdf).

### Libraries Used

- pandas
- numpy
- matplotlib
- wordcloud
- PIL
- plotly
- scikit-learn
- nltk
- seaborn
- requests
- streamlit

Make sure to have these libraries installed to run the project successfully. You can use `pip install <library-name>` to install any missing libraries.

---

Thank you for visiting my Sentiment Analysis project! For more details, check out the project code and documentation.

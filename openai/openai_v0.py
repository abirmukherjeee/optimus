#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openai

import numpy as np
import pandas as pd

import time

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from langdetect import detect


# In[2]:


def is_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return True
        else:
            return False
    except:
        return False


# In[3]:


# Download the required NLTK resources
nltk.download('vader_lexicon')

def analyze_sentiment_NLTK(review):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(review)['compound']
    
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"


# In[4]:


def analyze_sentiment_VADER(review):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(review)['compound']
    
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"


# In[5]:


def create_question(review):
    # Labels
    labels = ["Unable to login/access/registration",
              "Feature Request or Feedback",
              "Mobile Banking Feature Information",
              "Slowness/Crash/Loading/Bug/Glitch",
              "System Down or Temporary Issue",
              "Unable to download or update or install",
              "UI/UX interface issues/feedback",
              "Error while Operating or Using Mobile Banking - Loan",
              "Error while Operating or Using Mobile Banking - Credit Card",
              "Error while Operating or Using Mobile Banking - Fixed Deposit/Recurring Deposit/Savings accounts",
              "UPI Transaction issues",
              "Non UPI Transaction issues",
             "Others"]

    labels_str = "labels = " + str(labels).replace("'","") + ". "

    #review = "Can't view credit card on app. I need to have bank account to link accounts which I don't want to open."
    review = review
    review_str = "review = " + str(review)

    # Define your question
    question="Can you classify the review into only one of the label in the given list. " + str(labels_str) + str(review_str)

    #print(question)
    return(question)


# In[6]:


def get_response(question):
    # Set up OpenAI API credentials
    openai.api_key = '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'
    
    # Define the model and parameters
    model = "text-davinci-002"
    parameters = {
        "temperature": 0.6,
        "max_tokens": 50,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }

    # Generate a response from the model
    response = openai.Completion.create(
        engine=model,
        prompt=question,
        **parameters
    )

    # Extract the answer from the response
    answer = response.choices[0].text.strip()

    # Print the answer
    return(answer)


# In[7]:


def mandate_label(answer):
    try:
        # Labels
        labels = ["Unable to login/access/registration",
                  "Feature Request or Feedback",
                  "Mobile Banking Feature Information",
                  "Slowness/Crash/Loading/Bug/Glitch",
                  "System Down or Temporary Issue",
                  "Unable to download or update or install",
                  "UI/UX interface issues/feedback",
                  "Error while Operating or Using Mobile Banking - Loan",
                  "Error while Operating or Using Mobile Banking - Credit Card",
                  "Error while Operating or Using Mobile Banking - Fixed Deposit/Recurring Deposit/Savings account",
                  "Non UPI Transaction issue",
                  "UPI Transaction issue",
                 "Others"]
        
        for i in labels:
            if i.lower() in answer.lower():
                return(i)
            elif answer.lower() in i.lower():
                return(answer)
            
        if "Savings".lower() in answer.lower():
            return("Error while Operating or Using Mobile Banking - Fixed Deposit/Recurring Deposit/Savings account")
        
        elif "Non UPI".lower() in answer.lower():
            return("Non UPI Transaction issue")
        
        elif "UPI".lower() in answer.lower():
            return("UPI Transaction issue")
        
        elif "UI".lower() in answer.lower() or "UX".lower() in answer.lower():
            return("UI/UX interface issues/feedback")
        
        elif "Unable to login".lower() in answer.lower():
            return("Unable to login/access/registration")
        
        elif "Error while Operating or Using Mobile Banking".lower() in answer.lower():
            return("Error while Operating or Using Mobile Banking")
            
        return("Others")
            
    except Exception as e:
        return("Others")


# # Set up OpenAI API credentials
# openai.api_key = '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'

# In[8]:


def data_preprocess():
    playstore_reviews_df = pd.read_excel("resource/playstore.xlsx")
    appstore_reviews_df = pd.read_excel("resource/appstore.xlsx")
    
    playstore_reviews_df["source"] = "Google Playstore"
    appstore_reviews_df["source"] = "Apple Appstore"

    merged_reviews_df = pd.DataFrame({
        "source": pd.concat([playstore_reviews_df["source"], appstore_reviews_df["source"]], ignore_index=True),
        "username": pd.concat([playstore_reviews_df["userName"], appstore_reviews_df["userName"]], ignore_index=True),
        "datetime": pd.concat([playstore_reviews_df["at"], appstore_reviews_df["date"]], ignore_index=True),
        "rating": pd.concat([playstore_reviews_df["score"], appstore_reviews_df["rating"]], ignore_index=True),
        "review": pd.concat([playstore_reviews_df["content"], appstore_reviews_df["review"]], ignore_index=True),
        "sentiment": "", "classification": ""
    })
    
    merged_reviews_df = merged_reviews_df[merged_reviews_df["rating"] == 1]
    merged_reviews_df = merged_reviews_df.reset_index()
    merged_reviews_df = merged_reviews_df.drop("index", axis = 1)
    
    # Apply the language detection function to the 'text_column' and create a new column 'is_english'
    merged_reviews_df['is_english'] = merged_reviews_df['review'].apply(is_english)

    # Filter the DataFrame to keep only the rows with English strings
    merged_reviews_df = merged_reviews_df[merged_reviews_df['is_english'] == True]

    # Drop the 'is_english' column if desired
    merged_reviews_df.drop('is_english', axis=1, inplace=True)
    
    merged_reviews_df = merged_reviews_df[merged_reviews_df['review'].apply(lambda x: len(str(x).split()) > 5)]

    return(merged_reviews_df)


# In[9]:


def classify(df):
    for i, review in enumerate(df["review"]):
        if i < 6000:
            continue

        print(i, review)
        #print(get_response(create_question(review)))
        print()

        try:
            df.loc[i,"classification"] = get_response(create_question(review))
            df.loc[i,"hard_classification"] = mandate_label(df.loc[i,"classification"])
            df.loc[i,"sentiment"] = analyze_sentiment_VADER(review)
        except Exception as e:
            df.to_excel("result/analysed_reviews.xlsx", index = False)
            print(e)
            return(df)
        
    df.to_excel("result/analysed_reviews.xlsx", index = False)
    return(df)


# In[10]:


#df = data_preprocess()
df = pd.read_excel("result/analysed_reviews.xlsx")


# In[11]:


df = classify(df)


# text = "This is a Non UPI Transaction issue."
# mandate_label(text)

# In[ ]:





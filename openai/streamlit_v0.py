#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openai
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[2]:


def analyze_sentiment_VADER(review):
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(review)['compound']

        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"
    
    except Exception as e:
        return("System Resources Unavailable.")


# In[3]:


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


# In[4]:


def get_response(question):
    # Set up OpenAI API credentials
    openai.api_key = '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'
        
    try:
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
    
    except Exception as e:
        return("System Resources Unavailable.")


# In[9]:


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


# In[6]:


def classify(review):
        try:
            classification = get_response(create_question(review))
            sentiment = analyze_sentiment_VADER(review)
            hard_classification = mandate_label(classification)
            
            return(hard_classification, sentiment)
            
        except Exception as e:
            return("System Resources Unavailable.", "System Resources Unavailable.")


# In[12]:


review = "Yesterday I will QR code scanne to send money detect my account but but not credit another account"

print(classify(review))


# In[10]:


df = pd.read_excel("result/analysed_reviews.xlsx")

for i in range(len(df)):
    if "Other".lower() in df.loc[i,"hard_classification"].lower():
        df.loc[i,"hard_classification"] = mandate_label(df.loc[i,"classification"])
        
df.to_excel("result/optimised_reviews.xlsx", index = False)


# In[ ]:





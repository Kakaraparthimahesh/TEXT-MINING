# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 01:07:49 2022

@author: MAHESH
"""

# Performing sentimental analysis on the Elon-musk tweets (Exlon-musk.csv)

# IMPORTING REQUIRED LIBRIRIES 

import pandas as pd
import nltk
nltk.download()
from TextBlob import textblob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# data calling 

Elon_data = pd.read_csv("Elon_musk.csv",encoding= 'latin1')
Elon_data

list(Elon_data)
Elon_data.shape
Elon_data.info()
Elon_data.isnull().sum()

# Stage 1: Convert to Lower Text or cleaning the data

# Clean The Data

def cleantext(text):
    text = re.sub(r"@[A-Za-z0-9]+", "", text)       # Remove Mentions
    text = re.sub(r"#", "", text)                   # Remove Hashtags Symbol
    text = re.sub(r"RT[\s]+", "", text)             # Remove Retweets
    text = re.sub(r"https?:\/\/\S+", "", text)      # Remove The Hyper Link
    return text

# Clean The Text

Elon_data["Text"] = Elon_data["Text"].apply(cleantext)
Elon_data["Text"].head()

from textblob import TextBlob

# Get The Subjectivity

def sentiment_analysis(ds):
    sentiment = TextBlob(ds["Text"]).sentiment
    return pd.Series([sentiment.subjectivity, sentiment.polarity])

# Adding Subjectivity & Polarity

Elon_data[["subjectivity", "polarity"]] = Elon_data.apply(sentiment_analysis, axis=1)
Elon_data

allwords = " ".join([twts for twts in Elon_data["Text"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

# Compute The Negative, Neutral, Positive Analysis

def analysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

# Create a New Analysis Column

Elon_data["analysis"] = Elon_data["polarity"].apply(analysis)

Elon_data

positive_tweets = Elon_data[Elon_data['analysis'] == 'Positive']
negative_tweets = Elon_data[Elon_data['analysis'] == 'Negative']

print('positive tweets')
for i, row in positive_tweets[:5].iterrows():
  print(' -' + row['Text'])

print('negative tweets')
for i, row in negative_tweets[:5].iterrows():
  print(' -' + row['Text'])

plt.figure(figsize=(10, 8))

for i in range(0, Elon_data.shape[0]):
    plt.scatter(Elon_data["polarity"][i], Elon_data["subjectivity"][i], color = "Red")

plt.title("Sentiment Analysis")        # Add The Graph Title
plt.xlabel("Polarity")                 # Add The X-Label
plt.ylabel("Subjectivity")             # Add The Y-Label
plt.show() 

len(positive_tweets) 

len(negative_tweets)

len(positive_tweets) / len(negative_tweets)

# Since that number is positive, and quite high of a 
# ratio, we can also conclude that Elon is a positive guy.


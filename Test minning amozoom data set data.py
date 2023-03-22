# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 00:03:12 2022

@author: MAHESH
"""

import pandas as pd
import re
from textblob import TextBlob

# importing the data which has been scrapped from amazon website 

data=pd.read_csv('extract_reviews_test2.csv')
pd.set_option('display.max_colwidth', -1)

data.shape

list(data)

data.head

data.isnull().sum()

data

import nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

#---------------------
len(stopwords.words('english')) 
stop_words = set(stopwords.words('english'))

#Defining a dcitionary containing all the emojis and their meanings

emojis={':)':'smile',':-)':'smile',';d':'wink',':-E':'vampire',':(':'sad',
        ':-(':'sad',':-<':'sad',':P':'raspberry',':O':'surprised',
        ':-@':'shocked',':@':'shocked',':-$':'confused',':\\':'annoyed',
        ':#':'mute',':X':'mute',':^)':'smile',':-&':'confused','$_$':'greedy',
        '@@':'eyeroll',':-!':'confused',':-D':'smile',':-0':'yell','O.o':'confused',
        '<(-_-)>':'robot','d[-_-]b':'dj',":'-)":'sadsmile',';)':'wink',
        ';-)':'wink','O:-)':'angel','O*-)':'angel','(:-D':'gossip','=^.^=':'cat'}

#Defining a function to clean the data
def clean_text(kit):
    kit=str(kit).lower()
    kit=re.sub(r"@\S+",r'',kit)
    
    for i in emojis.keys():
        kit=kit.replace(i,emojis[i])
    kit=re.sub("\s+",' ',kit)
    kit=re.sub("\n",' ',kit)
    letters=re.sub('[^a-zA-Z]',' ',kit)
    return letters

#Defining a function to remove the stop words        

def stops_words(words):
    filter_words=[]
    for w in words:
        if w not in stop_words:
            filter_words.append(w)
    return filter_words

# Defining a function for sentiment analysis

def getSubjectivity(tex):
    return TextBlob(tex).sentiment.subjectivity

def getPolarity(tex):
    return TextBlob(tex).sentiment.polarity

def getAnalysis(score):
    if int(score)<0:
        return 'Negative'
    elif int(score)==0:
        return 'Neutral'
    elif int(score)>0:
        return 'Positive'

#Cleaning the data

data['comment']=data['comment'].apply(lambda x:clean_text(x))

#Removing stop words

data['comment']=data['comment'].apply(lambda x:x.split(" "))
data['comment']=data['comment'].apply(lambda x:stops_words(x))

#Stemming

from nltk.stem import PorterStemmer
stem=PorterStemmer()
data['comment']=data['comment'].apply(lambda x: [stem.stem(k) for k in x])

#Lemmatization
from nltk.stem import WordNetLemmatizer
lemm=WordNetLemmatizer()
data['comment']=data['comment'].apply(lambda x: [lemm.lemmatize(j) for j in x])

data['comment']=data['comment'].apply(lambda x: ' '.join(x))

#Preparing a target variable which shows the sentiment i.e, Subjectivity and Polarity
data['sentiment_subj']=data['comment'].apply(lambda x:getSubjectivity(x))
data['sentiment_subj'].describe()    

data['sentiment_pol']=data['comment'].apply(lambda x:getPolarity(x))
data['sentiment_pol'].describe()

sentiment=[]
for i in range(0,258,1):
    if data['sentiment_pol'].iloc[i,] < 0:
        sentiment.append('Negative')
    elif data['sentiment_pol'].iloc[i,] == 0:
        sentiment.append('Neutral')
    else:
        sentiment.append('Positive')

sentiment
Sentiment=pd.DataFrame(sentiment)
Sentiment.set_axis(['sentiment'],axis='columns',inplace=True)
data_new=pd.concat([data,Sentiment],axis=1)
data_new.shape

list(data_new)

#plotting

import seaborn as sns
sns.distplot(data_new['sentiment_subj'])
sns.distplot(data_new['sentiment_pol'])
sns.countplot(data_new['sentiment'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

%matplotlib inline
allwords = " ".join([twts for twts in data["comment"]])
wordCloud = WordCloud(width = 1000, height = 1000, random_state = 21, max_font_size = 119).generate(allwords)
plt.figure(figsize=(20, 20), dpi=80)
plt.imshow(wordCloud, interpolation = "bilinear")
plt.axis("off")
plt.show()

#Splitting into train and test
from sklearn.model_selection import train_test_split
data_train,data_test=train_test_split(data,test_size=0.25,random_state=41)

data_train_clean=data_train['comment']
data_test_clean=data_test['comment']

#Vectorize the data
from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer(use_idf=True)
data_train_clean=vector.fit_transform(data_train_clean)
data_test_clean=vector.transform(data_test_clean)

data_train_clean.toarray()
data_train_clean.toarray().shape

vector.get_feature_names()

len('Positive')

len('Negative')

len('Positive') / len('Negative')


# Inferences: It is a positive tweets.


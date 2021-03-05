import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Sentiment Analysis")
df=pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
x=df.iloc[:,10].values
y=df.iloc[:,1].values
processed_tweets=[]
for i in x:
  prc_tweet=re.sub(r'\W',' ',str(i))
  prc_tweet=re.sub(r'\s+[a-zA-Z]\s+',' ',prc_tweet)
  prc_tweet = re.sub(r'\^[a-zA-Z]\s+',' ', prc_tweet)
  prc_tweet= re.sub(r'\s+',' ', prc_tweet, flags=re.I)
  prc_tweet = re.sub(r'^b\s+','', prc_tweet)
  prc_tweet = prc_tweet.lower()
  processed_tweets.append(prc_tweet)
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.7, stop_words=stopwords.words('english'))
x=tfidfconverter.fit_transform(processed_tweets).toarray()
text_classifier=RandomForestClassifier(n_estimators=200, random_state=0)
st.title("Please wait , while we load the page to you")
text_classifier.fit(x,y)
select = st.text_input('Enter your message')
prc_tweet=re.sub(r'\W',' ',select)
prc_tweet=re.sub(r'\s+[a-zA-Z]\s+',' ',prc_tweet)
prc_tweet = re.sub(r'\^[a-zA-Z]\s+',' ', prc_tweet)
prc_tweet= re.sub(r'\s+',' ', prc_tweet, flags=re.I)
prc_tweet = re.sub(r'^b\s+','', prc_tweet)
prc_tweet = prc_tweet.lower()
tst=tfidfconverter.transform([prc_tweet]).toarray()
op=text_classifier.predict(tst)
st.title(op[0])

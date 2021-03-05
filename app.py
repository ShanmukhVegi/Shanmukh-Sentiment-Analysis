import streamlit as st
st.title("Sentiment Analysis")
import pandas as pd
import re
df=pd.read_csv("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")
x=df.iloc[:,10].values
y=df.iloc[:,1].values
processed_tweets=[]
for i in x:
  #removing all special characters
  prc_tweet=re.sub(r'\W',' ',str(i))

  #removing all single characters
  prc_tweet=re.sub(r'\s+[a-zA-Z]\s+',' ',prc_tweet)

  #removing single characters from the start
  prc_tweet = re.sub(r'\^[a-zA-Z]\s+',' ', prc_tweet)

  # Substituting multiple spaces with single space
  prc_tweet= re.sub(r'\s+',' ', prc_tweet, flags=re.I)

  # Removing prefixed 'b'
  prc_tweet = re.sub(r'^b\s+','', prc_tweet)

  # Converting to Lowercase
  prc_tweet = prc_tweet.lower()

  processed_tweets.append(prc_tweet)
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=3, max_df=0.7, stop_words=stopwords.words('english'))  
x= tfidfconverter.fit_transform(processed_tweets).toarray()
from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=300, random_state=0)  
text_classifier.fit(x, y)
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

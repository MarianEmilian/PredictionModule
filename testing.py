from utils import *
from numpy import sqrt
from datetime import date
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
import requests
import re

lematizer = WordNetLemmatizer()

unvector = get_articles(source = 'news.json')
vector = get_articles('news.json')
process_articles(vector)

vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
words = pickle.load(open('api/words.pickle','rb'))

import requests

BASE = "http://127.0.0.1:4200/sentiment"


for article in vector:
    if article['paragraphs']:
        response = requests.get(BASE, {'article' : article['paragraphs']})

        print(response.json()['sentiment'])



'''
article = vectorizer.transform([article['paragraphs']])
    index_value = {i[1]:i[0] for i in  vectorizer.vocabulary_.items()}
    content = {index_value[index]:value for (index,value) in zip(article.indices, article.data)}
    print(classify(content, words))
import requests

BASE = "http://127.0.0.1:4200/"

count = 0 
total = 0
import pickle

vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))

for article in unvector:
    text = ''
    for paragraph in article['paragraphs']:
        text = text + paragraph
    
    if text:
        total +=1
        response = requests.get(BASE + "/sentiment", {"text": text})
        print(response.json()['sentiment'], results[process_sentence(article['headline'])])'''

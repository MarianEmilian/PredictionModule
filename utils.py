
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import re
import spacy
import en_core_web_sm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
from math import e
from datetime import date
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lematizer = WordNetLemmatizer()



vectorizer = TfidfVectorizer(min_df= 3, max_df=0.3, stop_words= stopwords.words('english'))

import pickle

def get_articles(source):
    with open(source, 'r') as news:
        my_dict = json.load(news)
        return my_dict

def tf_idf(articles):
    content  = []
    
    for article in articles:
        content.append(article['paragraphs'])
    
    

    content = vectorizer.fit_transform(content)

    pickle.dump(vectorizer, open('models/vectorizer.sav', 'wb'))
    index_value = {i[1]:i[0] for i in  vectorizer.vocabulary_.items()}
    
    for row, article in zip(content, articles):
        article['paragraphs'] = {index_value[column]:value for (column,value) in zip(row.indices,row.data)}

def process_sentence(sentence):
    sentence.replace("\n", '')
    # special char ( !`~ etc)
    sentence = re.sub(r'\W', ' ', sentence)
    # single char (ex: what s banana -> what banana)
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    # single char start (ex: A banana -> banana)
    sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)
    # mult spaces to 1 space( ex:        ->  )
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def process_articles(articles):
    for article in articles:
        proc_paragraphs = ''
        # removes any unnecesarry characters from paragraphs
        for paragraph in article['paragraphs']:
            pp = process_sentence(sentence=paragraph)

            proc_paragraphs = proc_paragraphs + ' ' + pp
        article['paragraphs'] = proc_paragraphs


def create_dict(source):
    dict = {}
    with open(source) as words:
        lines = words.readlines()
        for line in lines:
            line.replace("\n", "")
            vals = line.split(" ")
            dict[vals[0]] = {
                "positive": float(vals[1]),
                "negative": float(vals[2])
            }
    pickle.dump(dict, open("dictionary/words.pickle", "wb"))
    return dict


def get_max_clicks(articles):
    # max clicks for each company (ex: max = {'tesla' : 1321 } 
    max_comp = 0
    for article in articles:
        if article['clicks'] > max_comp:
            max_comp = article['clicks']
    return max_comp

def norm_clicks(articles):
    max_comp = get_max_clicks(articles)    

    for article in articles:
        article['weighted_clicks'] = 0.5 + 0.5 * article['clicks'] / max_comp

def pos_neg_words(articles, words):
    
    for article in articles:
        pos_count, neg_count, total_count = 0, 0, 0 
        for word in word_tokenize(article['paragraphs']):
            if word in words:          
                if words[word]['positive'] > words[word]['negative']:
                    pos_count = pos_count + 1 
                elif words[word]['positive'] < words[word]['negative']:
                    neg_count = neg_count + 1
                total_count = total_count + 1
            elif lematizer.lemmatize(word) in words:
                 word = lematizer.lemmatize(word)
                 if words[word]['positive'] > words[word]['negative']:
                    pos_count = pos_count + 1 
                 elif words[word]['positive'] < words[word]['negative']:
                    neg_count = neg_count + 1
                 total_count = total_count + 1 
        article['pos_count'] = pos_count
        article['neg_count'] = neg_count
        article['total_count'] = total_count
    return articles


def get_mood(articles):
    pos_mood, neg_mood = 0,0
    for article in articles:
        days_passed = get_days(article['date'])
        time_comp = pow(e, -days_passed/10)
        if article['total_count']:
            pos_mood = pos_mood + article['pos_count'] * article['weighted_clicks'] / article['total_count'] * time_comp
            neg_mood = neg_mood + article['neg_count'] * article['weighted_clicks'] / article['total_count'] * time_comp
    return pos_mood, neg_mood

def get_month_index(month):
    months = ['filler', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for i in range(1,len(months)):
        if month == months[i]:
            return i
    return 1

def get_year_month_day(date):
    date = date.split(' ')
    year = int(date[2].replace(',',''))
    month = get_month_index(date[0].lower())
    day = int(date[1].replace(',',''))
    return year, month, day


def get_days(release_date):
    year, month, day = get_year_month_day(release_date)
    date_format = date(year, month, day)
    today = date.today()
    days = (today - date_format).days
    return days

def classify(article, words):
    result = 0
    for word in article:
        weight = article[word]
        
        if word in words:
            result = result + weight * words[word]['positive']
            result = result - weight * words[word]['negative']
    if result > 0:
        return 'positive', result
    else:
       return 'negative', result


def accurracy(results1, results):
    pos_corr, neg_corr = 0, 0 
    total_pos = 0
    total_neg = 0 

    for result in results1:
        pos_neg = ''
        if result['pos'] > result['neg']:
            pos_neg = 'positive'
            total_pos = total_pos + 1
        elif result['neg'] > result['pos']:
            pos_neg = 'negative'
            total_neg = total_neg + 1
        else:
            pos_neg = 'skip'

        if pos_neg == 'skip':
            if results[result['headline']] == 'negative':
                neg_corr = neg_corr + 1
                total_neg = total_neg + 1
            elif results[result['headline']] == 'positive':
                pos_corr = pos_corr + 1
                total_pos = total_pos + 1
        elif pos_neg == 'positive' and results[result['headline']] == 'positive':
            pos_corr = pos_corr + 1
        elif pos_neg == 'negative' and results[result['headline']] == 'negative':
            neg_corr = neg_corr + 1

    return (pos_corr + neg_corr ) * 100 / (total_neg + total_pos)


def vader_classifier(dict):
    cls = []
    for article in dict:
        content = ''
        for paragraph in article['paragraphs']:
            content = content + paragraph
        pol_score = SIA().polarity_scores(content)
        pol_score['headline'] = process_sentence(article['headline'])
        cls.append(pol_score)
    return cls

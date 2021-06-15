import pickle
import os
from flask import Flask, jsonify, make_response, request, redirect
import json
import sys
import pickle
from nltk.tokenize import word_tokenize

sys.path.append('D:\Licenta\PredIndex')
from PredictionModule.utils import get_mood, pos_neg_words, norm_clicks, process_articles, classify, process_sentence



WORDS = pickle.load(open("words.pickle", 'rb'))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

vectorizer = pickle.load(open('../models/vectorizer.sav', 'rb'))

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'GET':
        article = request.args.get('article')
        if article:
            article = article.split(', ')
            proc_paragraphs = ''
            for para in article:
                pp = process_sentence(sentence=para)
                proc_paragraphs = proc_paragraphs + ' ' + pp
            article = proc_paragraphs

            article_vector = vectorizer.transform([article])
            index_value = {i[1]:i[0] for i in  vectorizer.vocabulary_.items()}

            word_vector = {index_value[index]:value for (index,value) in zip(article_vector.indices, article_vector.data)}
            sentiment, score = classify(word_vector, WORDS)
    
            return make_response(jsonify({'sentiment': sentiment, 'score': score, 'status_code' : 200}), 200)
        return make_response(jsonify({'error':'sorry! unable to parse', 'status_code':500}), 500)


@app.route('/mood', methods=['GET', 'POST'])
def mood_analysis():
    if request.method == 'GET':
        articles = json.loads(request.args.get('articles'))
        if articles:
            for article in articles:
                article['paragraphs'] = article['paragraphs'].split(', ')
            
            process_articles(articles)
            norm_clicks(articles)
            pos_neg_words(articles, WORDS)
            pos, neg = get_mood(articles)
            
            return make_response(jsonify({'mood': (pos- neg) * 10 , 'status_code':200}), 200)
        return make_response(jsonify({'error':'sorry! unable to parse', 'status_code':500}), 500)


if __name__ == '__main__':
   app.run(port=4200, debug= True)
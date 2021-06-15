import pickle
import os
from flask import Flask, jsonify, make_response, request, redirect
import json
import sys
import pickle
from nltk.tokenize import word_tokenize

sys.path.append('D:\Licenta\PredIndex')
from PredictionModule.utils import get_mood, pos_neg_words, norm_clicks, process_articles


WORDS = pickle.load(open("words.pickle", 'rb'))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

vectorizer = pickle.load(open('../models/vectorizer.sav', 'rb'))

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'GET':
        text = request.args.get('text')
        if text:
            text_vector = vectorizer.transform([text])

            result = classifier.predict(text_vector)
            return make_response(jsonify({'sentiment': result[0], 'text': text, 'status_code':200}), 200)
        return make_response(jsonify({'error':'sorry! unable to parse', 'status_code':500}), 500)


@app.route('/mood', methods=['GET', 'POST'])
def mood_analysis():
    if request.method == 'GET':
        articles = json.loads(request.args.get('articles'))
        if articles:
            for article in articles:
                article['paragraphs'] = article['paragraphs'].split(', ')

                print(type(articles[0]['paragraphs']).__name__)
            
            process_articles(articles)
            norm_clicks(articles)
            pos_neg_words(articles, WORDS)
            pos, neg = get_mood(articles)
            
            return make_response(jsonify({'mood': (pos- neg) * 10 , 'status_code':200}), 200)
        return make_response(jsonify({'error':'sorry! unable to parse', 'status_code':500}), 500)


if __name__ == '__main__':
   app.run(port=4200, debug= True)
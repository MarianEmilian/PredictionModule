import pickle
import os
from flask import Flask, jsonify, make_response, request, redirect
from .utils import classify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))


@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'GET':
        text = request.args.get('text')
        if text:
            text_vector = vectorizer.transform([text])

            words = transf(text_vector, vectorizer.vocabulary_.items())

            result = classify(text_vector)
            return make_response(jsonify({'sentiment': result[0], 'text': text, 'status_code':200}), 200)
        return make_response(jsonify({'error':'sorry! unable to parse', 'status_code':500}), 500)

if __name__ == '__main__':
   app.run(port=4200)

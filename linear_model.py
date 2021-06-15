
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

trainData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/train.csv")


testData = pd.read_csv("https://raw.githubusercontent.com/Vasistareddy/sentiment_analysis/master/data/test.csv")


vectorizer = TfidfVectorizer(max_features=1000,min_df= 3, max_df=0.3, stop_words= stopwords.words('english'))


train_vectors = vectorizer.fit_transform(trainData['Content'])
test_vectors = vectorizer.transform(testData['Content'])


# Perform classification with SVM, kernel=linear


classifier_linear = svm.SVC(kernel='linear')


classifier_linear.fit(train_vectors, trainData['Label'])

prediction_linear = classifier_linear.predict(test_vectors)

# results

report = classification_report(testData['Label'], prediction_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])

import pickle
# pickling the vectorizer
pickle.dump(vectorizer, open('models/vectorizer.sav', 'wb'))
# pickling the model
pickle.dump(classifier_linear, open('models/svmclassifier.sav', 'wb'))
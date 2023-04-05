from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, confusion_matrix
import re
import string
import numpy as np
import pandas as pd

# Load the TfidfVectorizer model
transform_model = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

    #Load the LogisticRegression model
lr_model = pickle.load(open('lr.pkl', 'rb'))    

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['message']
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W"," ",text) 
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text) 
        text_vectorized = transform_model.transform([text])
        prediction = lr_model.predict(text_vectorized)[0]
        print(prediction)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)

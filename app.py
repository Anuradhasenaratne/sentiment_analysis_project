from flask import Flask, render_template, request, redirect
from helper import preprocessing, vectorizer, get_predticon
import pandas as pd

app = Flask(__name__)

# Load tokens globally
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

# Global variables
data = dict()
rev = []
positive = 0
negative = 0

@app.route("/")
def index():
    data['rev'] = rev
    data['positive'] = positive
    data['negative'] = negative
    return render_template('index.html', data=data)

@app.route("/", methods=['POST'])
def my_post():
    global positive, negative  # Declare global variables to modify them

    text = request.form['text']
    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt, tokens)  # Pass tokens to vectorizer
    prediction = get_predticon(vectorized_txt)

    # Update counts based on prediction
    if prediction == 'negative':
        negative += 1
    else:
        positive += 1    

    rev.insert(0, text)  # Add the new review to the beginning of the list

    # Update the data dictionary with the new values
    data['rev'] = rev
    data['positive'] = positive
    data['negative'] = negative

    # Return the updated data to the template
    return render_template('index.html', data=data)

if __name__ == "__main__":
    app.run(debug=True)

import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.stem import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Load the model and stopwords
with open('static/model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# Load vocabulary
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')  # Remove punctuation
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(word) for word in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(word.lower() for word in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].str.replace(r'\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(word for word in x.split() if word not in sw))
    
    # Return the preprocessed text
    return data["tweet"]

def vectorizer(ds, tokens):
    vectorized_1st = []
    
    for sentence in ds:
        sentence_1st = np.zeros(len(tokens))
        
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_1st[i] = 1
        
        vectorized_1st.append(sentence_1st)
    
    # Convert the list to a NumPy array
    vectorized_1st_new = np.asarray(vectorized_1st, dtype=np.float32)
    
    return vectorized_1st_new

def get_predticon(vectorized_txt):
    prediction = model.predict(vectorized_txt)
    
    # Assuming the model returns 0 for positive and 1 for negative
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'

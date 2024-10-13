import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps= PorterStemmer()

with open('static/model.pickle', 'rb')as f:
    model = pickle.load(f)

with open('static/model/corpora/stopwords/english', 'r') as file:
 sw=file.read().splitlines()

vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')  # Directly remove punctuation without adding space
    return text

def preprocessing(text):
    data =pd.DataFrame([text], columns=['tweet'])
    data["tweet"]= data["tweet"].apply(lambda x: " ".join(ps.stem(x)for x in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].str.replace(r'\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"]= data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    return data ["tweet"]


def vectorizer(ds, vocabulary):
    vectorized_1st = []

    for sentence in ds:
        sentence_1st = np.zeros(len(vocabulary))

        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                 sentence_1st[i] = 1
                
        vectorized_1st.append(sentence_1st)
        
    vectorized_1st_new = np.asarray(vectorized_1st, dtype=np.float32)

    return vectorized_1st_new

def get_prediction()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import models,layers,optimizers
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import bz2
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score
import re
import pickle

def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9])-1)
        texts.append(x[10:].strip())
    return np.array(labels),texts

train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')
test_labels,test_texts = get_labels_and_texts('test.ft.txt.bz2')

train_labels = train_labels[0:20000]
train_texts = train_texts[0:20000]

import re
NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)



from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, train_labels, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 0.75, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s"% (c, accuracy_score(y_val, lr.predict(X_val))))

print(lr.predict(X_test[29]))
print(test_labels[29])
with open("sentiment_model","wb") as f:
    pickle.dump(lr,f)
with open('count_v.pkl', 'wb') as fw:
    pickle.dump(cv.vocabulary_, fw)
sen = input('enter string')
dd = pd.DataFrame([[sen]],columns=['data'])
print(lr.predict(cv.transform(dd['data'])))
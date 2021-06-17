import pickle
import spacy
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import bz2
import re

#topic deciding
with open("model_pickle","rb") as f:
    g1 = pickle.load(f)
sentence = "The president greets the press in chicago."
print("topic decided is for ",sentence," is ",g1.predict([sentence]))

#similarity deciding
nlp=spacy.load('en_core_web_lg')
doc1 = nlp("The president greets the press in chicago.")
doc2 = nlp("Obama is speaks to the media in Illinois.")
print("similarity between sentence 1(The president greets the press in chicago.) and sentence 2(Obama is speaks to the media in Illinois.) is ")
print(doc1.similarity(doc2))
#sarcasm deciding
model1 = load_model('sarcasm_model1.h5')
model1.summary()
model2 = load_model('sarcasm_model2.h5')
model2.summary()
with open("sentiment_model","rb") as f:
    model3 = pickle.load(f)

vocab_size=3000
embedding_dim=32
max_len=32
trunc_type='post'
padding_type='post'
tokenizer= Tokenizer(num_words=vocab_size)

sent=["you broke my car , good job"]
seq=tokenizer.texts_to_sequences(sent)
padded=pad_sequences(seq,maxlen=max_len,padding=padding_type, truncating=trunc_type)
print(sent," from model1  ",model1.predict(padded))
print(sent," from model2  ",model2.predict(padded))

#sentiment deciding
#sen = input('enter string for sentiment ')
sen = "i am a good boy"
dd = pd.DataFrame([[sen]],columns=['data'])
with open("count_v.pkl","rb") as f:
    vec1 = pickle.load(f)
vec = CountVectorizer(decode_error="replace", vocabulary=vec1)
print("for " , sen,model3.predict(vec.transform(dd['data'])))

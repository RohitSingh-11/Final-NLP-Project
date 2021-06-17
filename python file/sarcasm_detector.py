import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding,Flatten,Dropout
from tensorflow import keras
from keras.callbacks import EarlyStopping
import pickle
from keras.models import load_model

df2 = pd.read_json("Sarcasm_Headlines_Dataset.json", 'r',lines=True)
dff2 = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", 'r',lines=True)
df =  pd.concat([df2, dff2])

def clean_text(text):
    text = text.lower()
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    text = text.lower()
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text


X=df['headline'].values
y=df['is_sarcastic'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

vocab_size=3000
embedding_dim=32
max_len=32
trunc_type='post'
padding_type='post'

tokenizer= Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)

training_sequences=tokenizer.texts_to_sequences(X_train)
training_padded=pad_sequences(training_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

testing_sequences=tokenizer.texts_to_sequences(X_test)
testing_padded=pad_sequences(testing_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

def create_model(vocabulary_size,embedding_dim,seq_len):
    model=Sequential()
    model.add(Embedding(vocabulary_size,embedding_dim,input_length=seq_len))
#     model.add(Flatten())
    model.add(LSTM(64,dropout=0.2,recurrent_dropout=0.25))
#     model.add(LSTM(64))
    model.add(Dense(1,activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.summary()
    return model

model=create_model(vocab_size+1,embedding_dim,max_len)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=7)
model.fit(training_padded,y_train,batch_size=64,epochs=15,verbose=2,validation_data=(testing_padded,y_test),callbacks=[es])
model.save("sarcasm_model1.h5")
def prediction_text(sent):
    sent=[sent]
    seq=tokenizer.texts_to_sequences(sent)
    padded=pad_sequences(seq,maxlen=max_len,padding=padding_type, truncating=trunc_type)
    return model.predict(padded)

sent="you broke my car , good job"
print(prediction_text(sent))

model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model2.add(Flatten())

model2.add(Dense(units=32,activation='relu'))
model2.add(Dropout(0.5))

model2.add(Dense(units=10,activation='relu'))
model2.add(Dropout(0.5))

model2.add(Dense(units=1,activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.01)
model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model2.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=7)
model2.fit(training_padded,y_train,batch_size=64,epochs=15,verbose=2,validation_data=(testing_padded,y_test),callbacks=[es])
model2.save("sarcasm_model2.h5")
def prediction_text2(sent):
    sent=[sent]
    seq=tokenizer.texts_to_sequences(sent)
    padded=pad_sequences(seq,maxlen=max_len,padding=padding_type, truncating=trunc_type)
    return model2.predict(padded)
sent="you broke my car , good job"
print(prediction_text2(sent))

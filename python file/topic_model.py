import pandas as pd
import numpy as np
import re
import string
import spacy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pickle

#nlp=spacy.load('en_core_web_lg')

df = pd.read_csv('combined1_csv.csv')
df.head()

# df.drop('article_link',axis=1,inplace=True)
def clean_text(text):
    text = text.lower()
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
#     emoji = re.compile("["
#                            u"\U0001F600-\U0001FFFF"  # emoticons
#                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
#                            "]+", flags=re.UNICODE)

#     text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text
new_text=[]
for text in df['data']:
      new_text.append(clean_text(text))
df['clean_text']=new_text

df['tidy_tweet']=df['clean_text'].str.replace("[^a-zA-Z#]"," ")

df['tidy_tweet']=df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


df.dropna(inplace=True)

blancks = []
for i,qn,x,y,b in df.itertuples():
    if type(b) == str:
        if b.isspace() or b=='':
            blancks.append(i)

df.drop(blancks,inplace=True)



tfidf = TfidfVectorizer(max_df=0.9,min_df=2,stop_words='english')

dtm = tfidf.fit_transform(df['tidy_tweet'])



nmf_model = NMF(n_components=3,random_state=42)

nmf_model.fit(dtm)

for i, arr in enumerate(nmf_model.components_):
    print(f"The words of NMF of high frequency of {i} is")
    print([tfidf.get_feature_names()[i] for i in arr.argsort()[-25:]])
#     print(arr[4])
    print('\n')

topic_sel = nmf_model.transform(dtm)


g_df = []
for j in range(0,3):
    op= []
    for i,data,cat,ct,tt in df.itertuples():
        if(topic_sel[i].argmax()==j):
            op.append(topic_sel[i].max())
    g_df.append(op)




x =range(0,len(g_df[0]))
y = g_df[0]
plt.scatter(x,y)
plt.show()

x =range(0,len(g_df[1]))
y = g_df[1]
plt.scatter(x,y)
plt.show()

x =range(0,len(g_df[2]))
y = g_df[2]
plt.scatter(x,y)
plt.show()


vals={0:'sports',1:'politics',2:'terrorism'}

df['topic']=topic_sel.argmax(axis=1)

df['category_decided']=df['topic'].map(vals)

print(accuracy_score(df['category'],df['category_decided']))
print(classification_report(df['category'],df['category_decided']))
print(confusion_matrix(df['category'],df['category_decided']))

x_train,x_test,y_train,y_test = train_test_split(df['data'],df['category_decided'],test_size=0.33,random_state=42)

tf_vect = TfidfVectorizer()

lin_svc = LinearSVC()
rt = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('lin_svc',LinearSVC())])
rt.fit(x_train,y_train)

op = rt.predict(x_test)

accuracy_score(op,y_test)
with open("model_pickle","wb") as f:
    pickle.dump(rt,f)
print(rt.predict(["hi i love to play football."]))

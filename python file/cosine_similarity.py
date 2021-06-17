from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import decimal as d
import pickle

sen2 ="i have baught a car"
sen1 ="a car was baught by me"
sen1 = sen1.lower()
sen2 = sen2.lower()
X_list = word_tokenize(sen1)
Y_list = word_tokenize(sen2)
sw = stopwords.words('english')
l1 =[];
l2 =[]
X_set = {w for w in X_list if not w in sw}
Y_set = {w for w in Y_list if not w in sw}
rvector = X_set.union(Y_set)
for w in rvector:
    if w in X_set: l1.append(1)
    else: l1.append(0)
    if w in Y_set: l2.append(1)
    else: l2.append(0)
c = 0
for i in range(len(rvector)):
        c+= l1[i]*l2[i]
cosine = (d.Decimal(c) / d.Decimal((sum(l1)*sum(l2))**0.5))*100;
with open("model_pickle","rb") as f:
    g1 = pickle.load(f)
if(g1.predict([sen1]) == g1.predict([sen2]) and cosine < 20):
    cosine = cosine + 25
elif(g1.predict([sen1]) == g1.predict([sen2]) and cosine < 50):
    cosine = cosine + 20;
elif(g1.predict([sen1]) == g1.predict([sen2]) and cosine < 70):
    cosine = cosine + 10;
elif(g1.predict([sen1]) == g1.predict([sen2]) and cosine < 90):
    cosine = cosine + 5;
print("similarity: ", cosine)

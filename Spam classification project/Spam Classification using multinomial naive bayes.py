import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
stopwords.words('english')
ps=PorterStemmer()
corpus=[]
spam_dataset=pd.read_csv('spam.csv',encoding='windows-1252',usecols=[0,1],names=['label','message'],header=0)
print(spam_dataset.head())
for i in range(len(spam_dataset)):
    review=re.sub('[^a-zA-Z\s]','',spam_dataset.iloc[i]['message'])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
tfidf=TfidfVectorizer()
x=tfidf.fit_transform(corpus)
y=pd.get_dummies(spam_dataset['label'])
y=y.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
spam_detect_model=MultinomialNB()
spam_detect_model.fit(x_train,y_train)
y_pred=spam_detect_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print(classification_report(y_test,y_pred))
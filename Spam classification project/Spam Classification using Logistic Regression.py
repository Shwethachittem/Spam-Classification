import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize Lemmatizer and corpus
lemmatizer = WordNetLemmatizer()
corpus = []

# Load dataset
spam_dataset = pd.read_csv('spam.csv', encoding='windows-1252', usecols=[0, 1], names=['label', 'message'], header=0)
print(spam_dataset.head())

# Text preprocessing
for i in range(len(spam_dataset)):
    review = re.sub('[^a-zA-Z\s]', '', spam_dataset.iloc[i]['message'])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

# Encode labels (ham = 0, spam = 1)
y = pd.get_dummies(spam_dataset['label'])
y = y.iloc[:, 0].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train, y_train)

# Predictions
y_pred = logreg_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

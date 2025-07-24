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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ps = WordNetLemmatizer()
corpus = []

# Load dataset
spam_dataset = pd.read_csv('spam.csv', encoding='windows-1252', usecols=[0,1], names=['label', 'message'], header=0)
print(spam_dataset.head())

# Text preprocessing
for i in range(len(spam_dataset)):
    review = re.sub('[^a-zA-Z\s]', '', spam_dataset.iloc[i]['message'])
    review = review.lower()
    review = review.split()
    review = [ps.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

# Encoding labels
y = pd.get_dummies(spam_dataset['label'])
y = y.iloc[:, 0].values  # Usually 'ham' as 0 and 'spam' as 1

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Spam Message Classification

This project demonstrates how to classify SMS messages as **Spam** or **Ham** (Not Spam) using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The models used include:

1. **Logistic Regression**
2. **Random Forest**
3. **Naive Bayes**

---

##  Dataset

The project uses the popular SMS Spam Collection Dataset (`spam.csv`), which contains labeled SMS messages as either `ham` (legitimate) or `spam`.

📎 [Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

| Label | Message |
|-------|---------|
| ham   | "Hey, are we still meeting today?" |
| spam  | "You’ve won a free iPhone! Click here to claim." |

---

## Techniques Used

### Text Preprocessing
- Lowercasing  
- Removing special characters  
- Tokenization  
- Stopword removal  
- Lemmatization

### Feature Extraction
- TF-IDF Vectorization

### Classification Models
- Logistic Regression  
- Random Forest Classifier  
- Multinomial Naive Bayes

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score

---

## Main Libraries
- `pandas`  
- `nltk`  
- `scikit-learn`

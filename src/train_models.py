import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

os.makedirs("outputs", exist_ok=True)

def train_all_models(df):
    X = df['cleaned_text'].values
    y = df['label'].values

    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)
    results['Logistic Regression'] = get_metrics(y_test, lr_pred)

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    results['Naive Bayes'] = get_metrics(y_test, nb_pred)

    # Passive Aggressive
    pa = PassiveAggressiveClassifier(max_iter=50, random_state=42, n_jobs=-1)
    pa.fit(X_train_tfidf, y_train)
    pa_pred = pa.predict(X_test_tfidf)
    results['Passive Aggressive'] = get_metrics(y_test, pa_pred)

    # Save models + vectorizer
    joblib.dump(lr, 'outputs/logistic_model.pkl')
    joblib.dump(nb, 'outputs/naive_bayes.pkl')
    joblib.dump(pa, 'outputs/passive_aggressive.pkl')
    joblib.dump(tfidf, 'outputs/tfidf_vectorizer.pkl')

    return results, tfidf, [lr, nb, pa], (X_test_tfidf, y_test)

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout

os.makedirs("outputs", exist_ok=True)

def get_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

def train_all_models(df):
    X = df['cleaned_text'].values
    y = df['label'].values

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

    results = {}

    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.8)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

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

    # Save traditional models
    joblib.dump(lr, 'outputs/logistic_model.pkl')
    joblib.dump(nb, 'outputs/naive_bayes.pkl')
    joblib.dump(pa, 'outputs/passive_aggressive.pkl')
    joblib.dump(tfidf, 'outputs/tfidf_vectorizer.pkl')

    max_words = 10000
    max_len = 300
    embedding_dim = 64

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')

    # --- CNN Model ---
    cnn_model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val),
                  epochs=4, batch_size=128, verbose=2)
    cnn_pred = (cnn_model.predict(X_test_seq) > 0.5).astype(int)
    results['CNN'] = get_metrics(y_test, cnn_pred)

    # --- LSTM Model ---
    lstm_model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val),
                   epochs=4, batch_size=128, verbose=2)
    lstm_pred = (lstm_model.predict(X_test_seq) > 0.5).astype(int)
    results['LSTM'] = get_metrics(y_test, lstm_pred)

    # Save deep learning models
    cnn_model.save('outputs/cnn_model.h5')
    lstm_model.save('outputs/lstm_model.h5')
    joblib.dump(tokenizer, 'outputs/tokenizer.pkl')

    return results

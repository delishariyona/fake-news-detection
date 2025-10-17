#  Fake News Detection using NLP

This project detects **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning / Deep Learning** techniques.  
It uses the **LIAR-style dataset** (`Fake.csv` + `True.csv`) to train and evaluate models that classify news articles as *real* or *fake*.

By combining traditional machine learning with deep learning architectures, this project provides a comprehensive comparative analysis of text classification performance.

---

##  Overview

The project involves:
- Cleaning and preprocessing raw news text using NLTK
- Converting text into numerical form using TF–IDF Vectorization and Word Embeddings
- Training and evaluating using Logistic Regression and LSTM 
- Measuring performance using accuracy, precision, recall, and F1-score

---

##  Models Implemented

This project evaluates five classification models:

1. **Logistic Regression** – a baseline linear classifier
It takes TF-IDF features (numerical representations of text based on word importance).
Each word feature is given a weight that shows how strongly it contributes to predicting the positive class.
It calculates a weighted sum of all features and passes it through a sigmoid function
If the probability ≥ 0.5 → class 1 (malicious), else class 0 (safe).
3. **Long Short-Term Memory (LSTM)**
Embedding Layer — Converts each word into a numerical vector capturing its meaning.
LSTM Layer — Reads the sequence of words, remembering context and relationships (e.g., “not attack”).
Dense Layer (ReLU) — Learns deeper patterns and combinations of features.
Dropout Layer — Prevents overfitting by randomly turning off neurons during training.
Output Layer (Sigmoid) — Produces a probability between 0 and 1, indicating the final prediction.

---

##  Dataset

- **Fake.csv** — contains fake news samples  
- **True.csv** — contains real news samples  
These datasets follow a LIAR-style format containing labeled text data.

The data is merged, shuffled, cleaned, and split into training and testing subsets before model training.

---

##  Setup Instructions 

Follow these steps to set up and run the project , upload the zip file of datset in the collab notebook.

it is a collab notebook in the fake news folder named FAKE_NEWS_DETECTION.ipynb 

run all the cells
 a quick Tunnel has been created for the gui , where you can check if the news is real or fake 

 refer the pdf the repo for detailed overview 




#  Fake News Detection using NLP

This project detects **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning / Deep Learning** techniques.  
It uses the **LIAR-style dataset** (`Fake.csv` + `True.csv`) to train and evaluate models that classify news articles as *real* or *fake*.

By combining traditional machine learning with deep learning architectures, this project provides a comprehensive comparative analysis of text classification performance.

---

##  Overview

The project involves:
- Cleaning and preprocessing raw news text using **NLTK**
- Converting text into numerical form using **TF–IDF Vectorization** and **Word Embeddings**
- Training and evaluating **five distinct classification models**
- Measuring performance using accuracy, precision, recall, and F1-score

---

##  Models Implemented

This project evaluates five classification models:

1. **Logistic Regression** – a baseline linear classifier  
2. **Naive Bayes** – probabilistic model suitable for text data  
3. **Passive Aggressive Classifier** – optimized for online learning  
4. **Convolutional Neural Network (CNN)** – deep learning model capturing spatial text patterns  
5. **Long Short-Term Memory (LSTM)** – recurrent neural network (RNN) capturing contextual dependencies in sequences  

---

##  Dataset

- **Fake.csv** — contains fake news samples  
- **True.csv** — contains real news samples  
These datasets follow a LIAR-style format containing labeled text data.

The data is merged, shuffled, cleaned, and split into training and testing subsets before model training.

---

##  Setup Instructions 

Follow these steps to set up and run the project �

```bash
#  Clone the repository
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

# Create and activate virtual environment
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate

#  Install dependencies
pip install -r requirements.txt

#  (Mac Users Only) – Fix SSL / NLTK download issues
python3
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
EXIT THE SHELL AFTER DOWNLOADING
RUN THE PROJECT USING python main.py

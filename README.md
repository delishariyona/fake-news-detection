#Fake News Detection using NLP

This project detects **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning** models.  
We use the **LIAR-style dataset** (`Fake.csv` + `True.csv`) to train and evaluate models that classify news as *real* or *fake*.

---

## Models Implemented

We compare and evaluate three ML models:

- **Logistic Regression**  
- **Multinomial Naive Bayes**  
- **Passive Aggressive Classifier**

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

python3 -m venv venv # creating virtual environment
source venv/bin/activate   # For macOS/Linux
venv\Scripts\activate      # For Windows
pip install -r requirements.txt ##install dependencies

## if you are a mac user If you face SSL certificate issues or NLTK download errors, open a Python shell and run:
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

##to run the project 
python main.py 

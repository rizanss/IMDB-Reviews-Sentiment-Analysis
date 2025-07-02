# Import Library
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
import pickle

# Inisialisasi FastAPI
app = FastAPI()

# Path Absolut ke Folder Model
MODEL_PATH = r"C:\Users\rizan\Documents\My Journey\Belajar AI Engineer\models"

# Load Logistic Regression Model
model_lr = pickle.load(open(os.path.join(MODEL_PATH, "logreg_model_sentiment.pkl"), "rb"))

# Load LSTM Model
model_lstm = tf.keras.models.load_model(os.path.join(MODEL_PATH, "lstm_model_sentiment.h5"))

# Load TF-IDF Vectorizer
with open(os.path.join(MODEL_PATH, "vectorizer_logreg_sentiment.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# Load Tokenizer
with open(os.path.join(MODEL_PATH, "tokenizer_lstm_sentiment.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# Batas maksimal panjang review
MAX_LEN = 200

# Kelas Pydantic buat request body
class ReviewRequest(BaseModel):
    review: str

# Endpoint buat tes API
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API with FastAPI!"}

# Endpoint buat prediksi Logistic Regression
@app.post("/predict/logistic")
def predict_logistic(data: ReviewRequest):
    review_tfidf = vectorizer.transform([data.review])
    prediction = model_lr.predict(review_tfidf)[0]
    sentiment = "Positif 😃" if prediction == 1 else "Negatif 😡"
    return {"review": data.review, "sentiment": sentiment}

# Endpoint buat prediksi LSTM
@app.post("/predict/lstm")
def predict_lstm(data: ReviewRequest):
    review_seq = tokenizer.texts_to_sequences([data.review])
    review_pad = pad_sequences(review_seq, maxlen=MAX_LEN, padding="post")
    prediction = model_lstm.predict(review_pad)[0][0]
    sentiment = "Positif 😃" if prediction > 0.5 else "Negatif 😡"
    return {"review": data.review, "sentiment": sentiment, "confidence": float(prediction)}

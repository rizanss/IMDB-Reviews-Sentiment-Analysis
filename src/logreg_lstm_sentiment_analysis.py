# IMPORT LIBRARY
import tensorflow as tf
import tensorflow_datasets as tfds
import re
import string
import nltk
import pandas as pd
import numpy as np
import gensim.downloader as api

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense

# DOWNLOAD DATASET IMDB
dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
print(f"Jumlah Data: {info.splits['train'].num_examples} training & {info.splits['test'].num_examples} testing")

# CEK CONTOH DATA
for text, label in dataset['train'].take(1):
    print("\n🎬 Review:", text.numpy().decode('utf-8'))
    print("🤖 Label Sentimen:", label.numpy())

# DOWNLOAD STOPWORDS
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

"""## Data Preprocessing"""

# FUNGSI PREPROCESSING TEKS
def clean_text(text):
    text = text.lower()  # Konversi ke string + lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Hapus punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Hapus stopwords
    return text

# CONTOH HASIL PREPROCESSING
for text, label in dataset['train'].take(1):
    cleaned_text = clean_text(text)
    print("\n🎬 Original Review:", text.numpy().decode('utf-8'))
    print("✂️ Cleaned Review:", cleaned_text)

# LOAD PRETRAINED WORD EMBEDDINGS (GLOVE)
glove_model = api.load("glove-wiki-gigaword-200")
print("\n Kata yang mirip dengan 'terrible':", glove_model.most_similar("terrible"))

# KONVERSI DATASET KE LIST
X_list = [x.numpy().decode("utf-8") for x, _ in dataset['train']]
y_list = [y.numpy() for _, y in dataset['train']]

# KONVERSI KE DATAFRAME
df = pd.DataFrame({'review': X_list, 'label': y_list})
df['clean_review'] = df['review'].apply(clean_text)

# TF-IDF FEATURE EXTRACTION
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['clean_review'])

"""## Data Splitting & Model Training"""

# SPLIT DATA TRAINING & TESTING
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)

"""### Logistic Regression Model"""

# TRAIN LOGISTIC REGRESSION MODEL
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# EVALUASI MODEL LOGISTIC REGRESSION
y_pred = model_lr.predict(X_test)
print(f"\n🎯 Akurasi Logistic Regression: {accuracy_score(y_test, y_pred):.4f}")
print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

"""### Long-Short Term Memory Model"""

# TOKENIZATION UNTUK LSTM
MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_review'])

X_seq = tokenizer.texts_to_sequences(df['clean_review'])
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# SPLIT DATA TRAINING & TESTING UNTUK LSTM
X_train, X_test, y_train, y_test = train_test_split(X_pad, df['label'], test_size=0.2, random_state=42)

# BUILD MODEL LSTM
model_lstm = keras.Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=50, input_length=MAX_LEN),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

# COMPILE & TRAIN LSTM
model_lstm.compile(loss="binary_crossentropy",
                   optimizer="adam",
                   metrics=["accuracy"])

history = model_lstm.fit(X_train,
                         y_train,
                         validation_data=(X_test, y_test),
                         epochs=5,
                         batch_size=32)

"""## Test & Evaluate Model"""

# EVALUASI MODEL LSTM
_, accuracy = model_lstm.evaluate(X_test, y_test)
print(f"\n🎯 Akurasi LSTM: {accuracy:.4f}")

# CONTOH REVIEW UNTUK PREDIKSI
new_reviews = [
    "This movie was absolutely amazing, I loved every second of it!",
    "What a waste of time, I regret watching this terrible film.",
    "The acting was decent but the story was very boring.",
    "One of the best movies I have ever seen!",
]

# PREDIKSI LOGISTIC REGRESSION
def predict_logistic_regression(reviews):
    reviews_tfidf = vectorizer.transform(reviews)
    predictions = model_lr.predict(reviews_tfidf)
    return ["Positif 😃" if pred == 1 else "Negatif 😡" for pred in predictions]

# PREDIKSI LSTM
def predict_lstm(reviews):
    reviews_seq = tokenizer.texts_to_sequences(reviews)
    reviews_pad = pad_sequences(reviews_seq, maxlen=MAX_LEN, padding="post")
    predictions = model_lstm.predict(reviews_pad)
    return ["Positif 😃" if pred > 0.5 else "Negatif 😡" for pred in predictions]

# TAMPILKAN HASIL PREDIKSI
logistic_results = predict_logistic_regression(new_reviews)
lstm_results = predict_lstm(new_reviews)

print("\n🎯 HASIL PREDIKSI SENTIMEN:")
for i, review in enumerate(new_reviews):
    print(f"\n🎬 Review: {review}")
    print(f"🤖 Logistic Regression: {logistic_results[i]}")
    print(f"🤖 LSTM: {lstm_results[i]}")

"""## Save Model"""

import pickle

# Simpan model Logistic Regression
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model_lr, f)

# Simpan TF-IDF Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Logistic Regression & TF-IDF Vectorizer berhasil disimpan!")

# Simpan model LSTM
model_lstm.save("lstm_model.h5")

# Simpan Tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model LSTM & Tokenizer berhasil disimpan!")

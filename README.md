# IMDB Reviews - Sentiment Analysis
🚀 Prediksi Sentimen Review Film menggunakan Logistic Regression & LSTM

## 📌 Fitur
- ✔️ Prediksi Sentimen dari Teks Review Film (Positif / Negatif)
- ✔️ Dua Model: Logistic Regression & LSTM
- ✔️ FastAPI untuk API yang Cepat & Scalable
- ✔️ Swagger UI untuk Testing API
- ✔️ Siap untuk Deployment di Cloud (Render / Railway)

## ⚙️ Instalasi
1️⃣ Clone Repository
```bash
git clone https://github.com/USERNAME/sentiment-analysis-api.git
cd sentiment-analysis-api
```

2️⃣ Buat Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## 🚀 Cara Menjalankan API
Jalankan API dengan FastAPI
```bash
uvicorn api_sentiment:app --reload
```
API akan running di ``http://127.0.0.1:8000``
Swagger UI bisa diakses di ``http://127.0.0.1:8000/docs``

## 📡 Deployment ke Cloud
### Opsi 1: Ngrok
1. Install ngrok
2. Jalankan API: ``uvicorn api_sentiment:app --host 0.0.0.0 --port 8000``
3. Start Ngrok: ``ngrok http 8000``

### Opsi 2: Render / Railway
1. Push project ke GitHub
2. Deploy di [Render](https://render.com) atau [Railway](https://railway.app)
3. Set Startup Command:
```bash
uvicorn api_sentiment:app --host 0.0.0.0 --port 8000
```

## 🎯 Contoh Request API
### 📌 Prediksi dengan Logistic Regression
Endpoint: ``POST /predict/logistic``
Request Body:
```json
{
  "review": "This movie is amazing!"
}
```
Response:
```json
{
  "review": "This movie is amazing!",
  "sentiment": "Positif 😃"
}
```

### 📌 Prediksi dengan LSTM
Endpoint: ``POST /predict/lstm``
Request Body:
```json
{
  "review": "This movie is so bad!"
}
```
Response:
```json
{
  "review": "This movie is so bad!",
  "sentiment": "Negatif 😡",
  "confidence": 0.1087
}
```

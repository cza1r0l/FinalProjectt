import os
import time
import re  # Fixes the 're' module error
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from io import StringIO
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.models import Sequential  # Fixed Sequential import
from keras.layers import LSTM, Dense, Embedding  # Fixed layer imports
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).parent
TEMPLATE_PATH = BASE_DIR / "templates" / "upload.html"

print(f"DEBUG: Checking template at {TEMPLATE_PATH}")
print(f"DEBUG: Exists? {TEMPLATE_PATH.exists()}")
print(f"DEBUG: Readable? {os.access(TEMPLATE_PATH, os.R_OK)}")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables to store results
model_results = []
current_dataset = None
vectorizer = None
mlb = None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


async def load_data(file) -> pd.DataFrame:
    content = await file.read()
    df = pd.read_json(StringIO(content.decode('utf-8')))
    game_data = []
    for game_id, game in df.items():
        if not game.get('genres') or not game.get('about_the_game'):
            continue
        genres = game['genres']
        desc = game['about_the_game']
        game_data.append({'description': desc, 'genres': genres})
    return pd.DataFrame(game_data)


def train_lstm(X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    start_time = time.time()
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64))
    model.add(LSTM(64))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred.round())
    return accuracy, time.time() - start_time


def train_naive_bayes(X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    start_time = time.time()
    model = OneVsRestClassifier(MultinomialNB())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, time.time() - start_time


def train_svm(X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    start_time = time.time()
    model = OneVsRestClassifier(LinearSVC())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, time.time() - start_time


def train_logistic_regression(X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    start_time = time.time()
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, time.time() - start_time


async def train_bert(X_train, y_train, X_test, y_test) -> Tuple[float, float]:
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=y_train.shape[1])

    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    ))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=2)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    ))

    loss, accuracy = model.evaluate(test_dataset)
    return accuracy, time.time() - start_time


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload")
async def force_upload():
    with open("templates/upload.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/train", response_class=HTMLResponse)
async def train_models(request: Request, file: UploadFile):
    global current_dataset, vectorizer, mlb, model_results
    df = await load_data(file)
    df['description'] = df['description'].apply(clean_text)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df['genres'])
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['description'])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_results = []

    lstm_acc, lstm_time = train_lstm(X_train, y_train, X_test, y_test)
    model_results.append({"name": "LSTM", "accuracy": lstm_acc, "time": lstm_time})

    nb_acc, nb_time = train_naive_bayes(X_train, y_train, X_test, y_test)
    model_results.append({"name": "Naive Bayes", "accuracy": nb_acc, "time": nb_time})

    svm_acc, svm_time = train_svm(X_train, y_train, X_test, y_test)
    model_results.append({"name": "SVM", "accuracy": svm_acc, "time": svm_time})
    lr_acc, lr_time = train_logistic_regression(X_train, y_train, X_test, y_test)
    model_results.append({"name": "Logistic Regression", "accuracy": lr_acc, "time": lr_time})
    try:
        bert_acc, bert_time = await train_bert(X_train, y_train, X_test, y_test)
        model_results.append({"name": "BERT", "accuracy": bert_acc, "time": bert_time})
    except Exception as e:
        print(f"BERT training failed: {e}")
    model_results.sort(key=lambda x: x['accuracy'], reverse=True)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "results": model_results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
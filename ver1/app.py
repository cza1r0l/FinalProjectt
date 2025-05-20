import json
import pandas as pd
import numpy as np
import re
import joblib
import os
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

BASE_DIR = Path(__file__).resolve().parent

def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    game_data = []
    for game_id, game in data.items():
        if not game.get('genres') or not game.get('about_the_game'):
            continue
        genres = game['genres']
        desc = game['about_the_game']
        game_data.append({'description': desc, 'genres': genres})

    return pd.DataFrame(game_data)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def train_model(data_path):
    df = load_data(data_path)
    df['description'] = df['description'].apply(clean_text)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df['genres'])

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['description'])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # Save models
    joblib.dump(model, BASE_DIR / 'best_model.joblib')
    joblib.dump(vectorizer, BASE_DIR / 'tfidf_vectorizer.joblib')
    joblib.dump(mlb, BASE_DIR / 'label_binarizer.joblib')


train_model(BASE_DIR / 'games.json')

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

TEMPLATES_DIR = BASE_DIR / "templates"
if not TEMPLATES_DIR.exists():
    os.makedirs(TEMPLATES_DIR)
    print(f"Created templates directory at {TEMPLATES_DIR}")

try:
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
except Exception as e:
    print(f"Jinja2 templates error: {e}")
    raise

try:
    model = joblib.load(BASE_DIR / "best_model.joblib")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.joblib")
    mlb = joblib.load(BASE_DIR / "label_binarizer.joblib")
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    MODELS_LOADED = False


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_genre(request: Request, description: str = Form(...)):
    if not MODELS_LOADED:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "Prediction models not loaded. Please train the model first."
        })

    try:
        desc_cleaned = clean_text(description)
        vectorized = vectorizer.transform([desc_cleaned])
        prediction = model.predict(vectorized)
        genres = mlb.inverse_transform(prediction)
        return templates.TemplateResponse("results.html", {
            "request": request,
            "description": description,
            "genres": genres[0] if genres else []
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Prediction error: {str(e)}"
        })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
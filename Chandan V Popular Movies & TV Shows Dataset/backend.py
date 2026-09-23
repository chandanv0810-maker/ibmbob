from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "streaming_content_trends.csv"
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"

app = Flask(__name__)


def load_catalog() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["original_language"] = data["original_language"].fillna("Unknown")
    data["search_category"] = data["search_category"].fillna("unknown")
    data["media_type"] = data["media_type"].fillna("unknown")
    return data


def feature_frame(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "media_type": payload.get("media_type", "movie"),
        "original_language": payload.get("original_language", "en"),
        "search_category": payload.get("search_category", "popular"),
        "release_year": payload.get("release_year", 2024),
        "vote_count": payload.get("vote_count", 500),
        "popularity": payload.get("popularity", 50),
    }])


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.get("/api/model-info")
def model_info():
    metrics = pd.read_json(METRICS_PATH, typ="series").to_dict() if METRICS_PATH.exists() else {"status": "model not trained"}
    return jsonify(metrics)


@app.get("/api/dataset-summary")
def dataset_summary():
    data = load_catalog()
    return jsonify({"rows": int(len(data)), "columns": list(data.columns), "media_types": data["media_type"].value_counts().to_dict(), "languages": data["original_language"].value_counts().head(10).to_dict(), "year_range": [int(data["release_year"].min()), int(data["release_year"].max())]})


@app.post("/api/predict")
def predict():
    if not MODEL_PATH.exists():
        return jsonify({"error": "Model not found. Run train_model.py first."}), 503
    payload = request.get_json(silent=True) or {}
    model = joblib.load(MODEL_PATH)
    prediction = float(model.predict(feature_frame(payload))[0])
    return jsonify({"predicted_rating": round(max(0, min(10, prediction)), 3), "inputs": payload})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
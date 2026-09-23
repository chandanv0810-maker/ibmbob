from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "streaming_content_trends.csv"
MODEL_PATH = ROOT / "model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
PREDICTIONS_PATH = ROOT / "model_predictions.csv"


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data[["media_type", "original_language", "search_category", "release_year", "vote_count", "popularity"]].copy()
    features["release_year"] = features["release_year"].fillna(features["release_year"].median())
    return features


def train() -> dict:
    data = pd.read_csv(DATA_PATH)
    data["original_language"] = data["original_language"].fillna("Unknown")
    data["search_category"] = data["search_category"].fillna("unknown")
    data["media_type"] = data["media_type"].fillna("unknown")

    features = build_features(data)
    target = data["vote_average"].astype(float)
    categorical = ["media_type", "original_language", "search_category"]
    numeric = ["release_year", "vote_count", "popularity"]
    preprocess = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), numeric),
         ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)]
    )
    model = Pipeline([("preprocess", preprocess), ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1))])

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {"rows": int(len(data)), "features": list(features.columns), "mae": round(float(mean_absolute_error(y_test, predictions)), 4), "r2": round(float(r2_score(y_test, predictions)), 4)}

    full_predictions = model.predict(features)
    output = data[["id", "title", "media_type", "vote_average"]].copy()
    output["predicted_rating"] = full_predictions.round(3)
    output["prediction_error"] = (output["predicted_rating"] - output["vote_average"]).round(3)
    output.to_csv(PREDICTIONS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(train())
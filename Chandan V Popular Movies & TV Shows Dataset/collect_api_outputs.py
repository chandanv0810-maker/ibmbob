import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "api_outputs"
BASE_URL = "http://127.0.0.1:5000"


def collect() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    responses = {
        "health.json": requests.get(f"{BASE_URL}/health", timeout=5).json(),
        "model_info.json": requests.get(f"{BASE_URL}/api/model-info", timeout=5).json(),
        "dataset_summary.json": requests.get(f"{BASE_URL}/api/dataset-summary", timeout=5).json(),
        "prediction.json": requests.post(f"{BASE_URL}/api/predict", json={"media_type": "movie", "original_language": "en", "search_category": "popular", "release_year": 2025, "vote_count": 1500, "popularity": 125}, timeout=5).json(),
    }
    for filename, payload in responses.items():
        (OUTPUT_DIR / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {len(responses)} API responses to {OUTPUT_DIR}")


if __name__ == "__main__":
    collect()
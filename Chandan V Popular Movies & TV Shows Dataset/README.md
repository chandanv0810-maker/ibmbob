# StreamScope

StreamScope is a Python-powered Streamlit dashboard for exploring `streaming_content_trends.csv`. It turns the catalog into an interactive content intelligence view: filter the catalog, compare movie and TV signals, find high-momentum titles, and inspect genre, language, rating, and release-year patterns.

## Run locally

From this folder:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The same frontend can also be launched with `streamlit run ui.py`.

The app reads the CSV from the same folder as `app.py`, so the data file should remain beside the application.

## Project plan

1. **Ingest**: load and normalize dates, years, categories, genres, and missing values with pandas.
2. **Filter**: let users narrow by format, discovery signal, language, release year, rating, and recency.
3. **Analyze**: show catalog volume, surfacing signals, popularity versus rating, genre density, and language mix.
4. **Browse**: provide a searchable shortlist sorted by popularity for quick title discovery.
5. **Extend**: add watch-provider data, regional breakdowns, or a recommendation model when those fields are available.

## Stack

- **Backend and data layer**: Python, pandas
- **Frontend**: Streamlit
- **Interactive charts**: Altair
- **Prediction API**: Flask
- **Modeling**: scikit-learn Random Forest regression

## Full project workflow

```powershell
python train_model.py
python generate_report.py
python backend.py
```

With the Flask server running in another terminal, collect the API artifacts:

```powershell
python collect_api_outputs.py
```

This creates `model.joblib`, `model_metrics.json`, `model_predictions.csv`, `project_report.md`, `report_assets/top_genres.png`, and JSON responses in `api_outputs/`.
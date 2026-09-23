from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report_assets"


def generate() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / "streaming_content_trends.csv")
    data["genres"] = data["genres"].fillna("Unknown")
    genre_counts = data.assign(genre=data["genres"].str.split(", ")).explode("genre")["genre"].value_counts().head(10).sort_values()
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    genre_counts.plot.barh(ax=ax, color="#1fbf9f")
    ax.set_title("Top genres in the streaming catalog")
    ax.set_xlabel("Titles")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "top_genres.png", dpi=160)
    plt.close(fig)

    format_counts = data["media_type"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor("#f7f8f4")
    axes[0].set_facecolor("#f7f8f4")
    axes[1].set_facecolor("#f7f8f4")
    axes[0].bar(["Titles", "Avg rating", "Avg popularity"], [len(data), data["vote_average"].mean(), data["popularity"].mean()], color=["#1fbf9f", "#f26b5e", "#18222d"])
    axes[0].set_title("Catalog pulse", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="x", visible=False)
    axes[1].pie(format_counts.values, labels=format_counts.index, colors=["#1fbf9f", "#f26b5e"], startangle=90, wedgeprops={"width": 0.42, "edgecolor": "#f7f8f4"})
    axes[1].set_title("Format mix", loc="left", fontweight="bold")
    fig.suptitle("STREAMSCOPE  /  CONTENT INTELLIGENCE", x=0.06, ha="left", fontsize=16, fontweight="bold", color="#18222d")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(REPORT_DIR / "ui_preview.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    report = f"""# StreamScope Project Report

## Objective

StreamScope turns `streaming_content_trends.csv` into an interactive catalog intelligence product with a Streamlit frontend and a Flask prediction API.

## Dataset

- Rows: **{len(data):,}**
- Columns: **{len(data.columns)}**
- Formats: **{data['media_type'].value_counts().to_dict()}**
- Release years: **{int(data['release_year'].min())} to {int(data['release_year'].max())}**
- Mean audience rating: **{data['vote_average'].mean():.2f} / 10**

## Architecture

1. `train_model.py` cleans the catalog and trains a Random Forest regression model for audience rating.
2. `backend.py` serves health, model information, dataset summary, and prediction endpoints.
3. `ui.py` launches the Streamlit frontend from the existing `app.py` dashboard.
4. `collect_api_outputs.py` captures API responses under `api_outputs/`.
5. `generate_report.py` creates this report asset and a genre chart under `report_assets/`.

## Model evaluation

The training script writes `model_metrics.json` with holdout MAE and R-squared values. The model is intended as a demonstration of the project pipeline, not as a production recommendation system.

## Generated visual

![Top genres](report_assets/top_genres.png)

![UI preview](report_assets/ui_preview.png)
"""
    (ROOT / "project_report.md").write_text(report, encoding="utf-8")
    print("Generated project_report.md and report_assets/top_genres.png")


if __name__ == "__main__":
    generate()
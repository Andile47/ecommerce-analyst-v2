# E-Commerce Sentiment Analysis Dashboard

A Streamlit dashboard for exploring customer review sentiment with VADER. It includes sentiment KPIs and charts, a searchable and filterable review explorer, a single-review analyzer, and evidence-based review insights.

## Run locally

```bash
python -m venv .venv
```

Activate the environment, then install dependencies and launch the dashboard:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the dashboard at [http://localhost:8501](http://localhost:8501). If that port is already in use, use the alternate localhost URL printed by Streamlit. The current running instance is at [http://localhost:8507](http://localhost:8507).

The app uses `sample_reviews.csv` by default when that file is present in the project root or `data/` directory.

## Uploading data

Upload a CSV containing a `review` column. Optional date, product, and category columns enable additional filtering and comparisons. Original source columns are preserved in review details and filtered CSV exports.

Sentiment is calculated locally with VADER's compound score: Positive at `>= 0.05`, Negative at `<= -0.05`, and Neutral between those thresholds. The compound score is polarity, not a probability or confidence estimate. The first run may download NLTK's VADER lexicon if it is not already installed.

## Dependencies

See `requirements.txt` for Streamlit, Pandas, Plotly, and NLTK.

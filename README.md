# Movie Recommendation System

A movie recommender trained on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset (`ml-latest-small`:
~100,000 ratings from 600 users on ~9,700 movies) — the same style of data Netflix-era
recommender research was built on.

Two complementary approaches are implemented:

- **Collaborative filtering** (`src/train_cf.py`) — matrix factorization (SVD) over the
  user-item rating matrix. Learns from patterns like "users who rated similarly to you
  also liked X." Powers personalized "recommended for you" style suggestions.
- **Content-based filtering** (`src/content_based.py`) — TF-IDF over movie genres/tags
  with cosine similarity. Powers "because you watched X, you might like Y" style
  suggestions, and works even for movies with few ratings (no cold-start problem).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m src.data           # downloads and caches the MovieLens dataset
```

## Usage

Train and evaluate the collaborative filtering model:

```bash
python -m src.train_cf
```

Try content-based similarity search:

```bash
python -m src.content_based
```

Launch the interactive demo:

```bash
python app.py
```

## Project structure

```
src/
  data.py            # downloads/loads MovieLens ratings + movie metadata
  train_cf.py        # SVD-based collaborative filtering, evaluates RMSE
  content_based.py   # TF-IDF + cosine similarity over genres
  evaluate.py         # shared metrics
app.py               # Gradio demo
```

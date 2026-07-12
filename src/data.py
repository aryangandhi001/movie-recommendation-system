"""Downloads and loads the MovieLens ml-latest-small dataset.

Source: https://grouplens.org/datasets/movielens/
~100,000 ratings from 610 users across 9,742 movies.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RAW_DIR = Path("data/raw")
EXTRACTED_DIR = RAW_DIR / "ml-latest-small"


def _download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MovieLens dataset from {DATASET_URL} ...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted to {EXTRACTED_DIR}")


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ratings, movies) DataFrames.

    ratings columns: userId, movieId, rating, timestamp
    movies columns:  movieId, title, genres (pipe-separated, e.g. "Action|Adventure")
    """
    if not EXTRACTED_DIR.exists():
        _download_and_extract()

    ratings = pd.read_csv(EXTRACTED_DIR / "ratings.csv")
    movies = pd.read_csv(EXTRACTED_DIR / "movies.csv")
    return ratings, movies


if __name__ == "__main__":
    ratings, movies = load_movielens()
    print(f"Ratings: {len(ratings):,} rows, {ratings['userId'].nunique():,} users, "
          f"{ratings['movieId'].nunique():,} rated movies")
    print(f"Movies: {len(movies):,} rows")
    print(ratings.head())
    print(movies.head())

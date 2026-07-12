"""Content-based recommender over TMDB titles: TF-IDF (genres + overview text)
+ cosine similarity, with a popularity/rating-weighted "trending" ranking.

Note: TMDB doesn't expose individual user rating histories, so true collaborative
filtering ("users like you also liked...") isn't possible on this data. This
recommender instead combines content similarity with a popularity signal.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as l2_normalize

from src.tmdb_data import load_titles

MODEL_PATH = Path("models/content_model.joblib")
MIN_VOTE_COUNT = 20  # filter out obscure titles with too few votes to trust vote_average


def _weighted_rating(df: pd.DataFrame, m: int = MIN_VOTE_COUNT) -> pd.Series:
    """IMDB-style Bayesian weighted rating: pulls low-vote-count titles toward
    the global mean so a 10.0 from 3 votes doesn't outrank a 8.5 from 5,000."""
    C = df["vote_average"].mean()
    v = df["vote_count"]
    R = df["vote_average"]
    return (v / (v + m)) * R + (m / (v + m)) * C


def build():
    titles = load_titles()
    titles = titles[titles["vote_count"] >= 5].reset_index(drop=True)
    titles["weighted_rating"] = _weighted_rating(titles)

    text = (
        titles["genres"].str.replace("|", " ", regex=False).fillna("")
        + " " + titles["overview"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20_000)
    vectors = vectorizer.fit_transform(text)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(vectors)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "nn": nn, "vectors": vectors, "titles": titles},
        MODEL_PATH,
    )
    print(f"Saved content-based index to {MODEL_PATH} ({len(titles):,} titles)")


class ContentRecommender:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            build()
        state = joblib.load(model_path)
        self.nn = state["nn"]
        self.vectors = state["vectors"]
        self.titles = state["titles"]
        self._label_to_index = {
            self._label(row): i for i, row in self.titles.iterrows()
        }
        self._tmdb_id_to_index = {
            (row["tmdb_id"], row["media_type"]): i for i, row in self.titles.iterrows()
        }
        wr = self.titles["weighted_rating"].to_numpy()
        wr_range = wr.max() - wr.min()
        self._trending_norm = (wr - wr.min()) / wr_range if wr_range > 0 else np.zeros_like(wr)

    @staticmethod
    def _label(row) -> str:
        kind = "Movie" if row["media_type"] == "movie" else "TV"
        return f"{row['title']} ({row['year']}) [{kind}]"

    def all_labels(self) -> list[str]:
        return list(self._label_to_index.keys())

    def trending(self, k: int = 10, media_type: str | None = None) -> pd.DataFrame:
        df = self.titles
        if media_type:
            df = df[df["media_type"] == media_type]
        return df.sort_values("weighted_rating", ascending=False).head(k)

    def recommend(self, label: str, k: int = 10):
        if label not in self._label_to_index:
            return []
        idx = self._label_to_index[label]
        distances, indices = self.nn.kneighbors(self.vectors[idx], n_neighbors=k + 1)
        results = []
        for dist, i in zip(distances[0], indices[0]):
            if i == idx:
                continue
            row = self.titles.iloc[i]
            results.append((self._label(row), row["genres"], 1 - float(dist)))
        return results[:k]

    def recommend_for_profile(self, liked_labels: list[str], k: int = 10, content_weight: float = 0.7):
        """Personalized recommendations from a small set of titles a user says they
        like: builds a taste-profile vector (mean of their TF-IDF vectors) and ranks
        the catalog against it, blended with the trending/quality signal so results
        favor well-regarded titles, not just the narrowest genre match."""
        liked_indices = [self._label_to_index[l] for l in liked_labels if l in self._label_to_index]
        if not liked_indices:
            return []

        profile_vector = np.asarray(self.vectors[liked_indices].mean(axis=0))
        profile_vector = l2_normalize(profile_vector)
        content_scores = np.asarray(self.vectors @ profile_vector.T).flatten()

        final_scores = content_weight * content_scores + (1 - content_weight) * self._trending_norm

        ranked = np.argsort(-final_scores)
        results = []
        for i in ranked:
            if i in liked_indices:
                continue
            row = self.titles.iloc[i]
            results.append((self._label(row), row["genres"], float(final_scores[i])))
            if len(results) >= k:
                break
        return results

    def label_for_tmdb_id(self, tmdb_id: int, media_type: str = "movie") -> str | None:
        idx = self._tmdb_id_to_index.get((tmdb_id, media_type))
        if idx is None:
            return None
        return self._label(self.titles.iloc[idx])


if __name__ == "__main__":
    build()
    rec = ContentRecommender()
    sample = rec.all_labels()[0]
    print(f"Because you watched: {sample}")
    for label, genres, sim in rec.recommend(sample):
        print(f"  {sim:.3f}  {label}  [{genres}]")

    print("\nTop trending overall:")
    for _, row in rec.trending(10).iterrows():
        print(f"  {row['weighted_rating']:.2f}  {rec._label(row)}")

    liked = rec.all_labels()[:3]
    print(f"\nTaste-profile recommendations for someone who likes: {liked}")
    for label, genres, score in rec.recommend_for_profile(liked):
        print(f"  {score:.3f}  {label}  [{genres}]")

"""Content-based recommender: TF-IDF over movie genres + cosine similarity.

Unlike collaborative filtering, this doesn't need any rating history for a movie,
so it works even for brand-new or rarely-rated titles ("because you watched X...").
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.data import load_movielens

MODEL_PATH = Path("models/content_model.joblib")


def build():
    _, movies = load_movielens()
    genre_text = movies["genres"].str.replace("|", " ", regex=False)

    vectorizer = TfidfVectorizer()
    genre_vectors = vectorizer.fit_transform(genre_text)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(genre_vectors)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "nn": nn, "genre_vectors": genre_vectors, "movies": movies},
        MODEL_PATH,
    )
    print(f"Saved content-based index to {MODEL_PATH} ({len(movies):,} movies)")


class ContentRecommender:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            build()
        state = joblib.load(model_path)
        self.nn = state["nn"]
        self.genre_vectors = state["genre_vectors"]
        self.movies = state["movies"]
        self.title_to_index = {t: i for i, t in enumerate(self.movies["title"])}

    def find_titles(self, query: str, limit: int = 10):
        query = query.lower()
        matches = self.movies[self.movies["title"].str.lower().str.contains(query, regex=False)]
        return matches["title"].tolist()[:limit]

    def recommend(self, title: str, k: int = 10):
        if title not in self.title_to_index:
            return []
        idx = self.title_to_index[title]
        distances, indices = self.nn.kneighbors(self.genre_vectors[idx], n_neighbors=k + 1)
        results = []
        for dist, i in zip(distances[0], indices[0]):
            if i == idx:
                continue
            row = self.movies.iloc[i]
            results.append((row["title"], row["genres"], 1 - float(dist)))
        return results[:k]


if __name__ == "__main__":
    build()
    rec = ContentRecommender()
    sample = rec.movies.iloc[0]["title"]
    print(f"Because you watched: {sample}")
    for title, genres, sim in rec.recommend(sample):
        print(f"  {sim:.3f}  {title}  [{genres}]")

"""Classic collaborative filtering (SVD matrix factorization) trained on real
MovieLens user ratings, mapped onto our TMDB catalog via MovieLens' own
movieId -> tmdbId links table.

This is deliberately kept separate from the TMDB-based content/personalization
system: it demonstrates genuine "users like you also liked..." CF using real
historical rating data, which TMDB itself doesn't expose. Coverage is limited
to titles released before ~2018 (MovieLens ratings stop there) that also
happen to be in our 2005-2025 TMDB catalog.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

from src.evaluate import rmse

ML_DIR = Path("data/raw/ml-latest-small")
MODEL_PATH = Path("models/movielens_cf.npz")
N_FACTORS = 50


def _load_ratings_and_links():
    ratings = pd.read_csv(ML_DIR / "ratings.csv")
    links = pd.read_csv(ML_DIR / "links.csv")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)
    return ratings, links


def train(n_factors: int = N_FACTORS):
    ratings, links = _load_ratings_and_links()
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    movie_index = {m: i for i, m in enumerate(movie_ids)}

    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    rows = train_ratings["userId"].map(user_index)
    cols = train_ratings["movieId"].map(movie_index)
    matrix = coo_matrix(
        (train_ratings["rating"].values, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    ).tocsr()

    user_means = np.asarray(matrix.sum(axis=1)).flatten()
    counts = np.diff(matrix.indptr)
    counts[counts == 0] = 1
    user_means = user_means / counts

    matrix_centered = matrix.tolil()
    for u_idx in range(matrix.shape[0]):
        row = matrix.getrow(u_idx)
        if row.nnz > 0:
            matrix_centered.rows[u_idx] = row.indices.tolist()
            matrix_centered.data[u_idx] = (row.data - user_means[u_idx]).tolist()
    matrix_centered = matrix_centered.tocsr().asfptype()

    k = min(n_factors, min(matrix_centered.shape) - 1)
    U, sigma, Vt = svds(matrix_centered, k=k)
    sigma = np.diag(sigma)
    user_factors = U @ sigma
    item_factors = Vt.T

    def predict(u_id, m_id):
        if u_id not in user_index or m_id not in movie_index:
            return float(user_means.mean())
        u_idx, m_idx = user_index[u_id], movie_index[m_id]
        return float(user_means[u_idx] + user_factors[u_idx] @ item_factors[m_idx])

    test_preds = [predict(u, m) for u, m in zip(test_ratings["userId"], test_ratings["movieId"])]
    test_rmse = rmse(test_ratings["rating"].values, test_preds)
    print(f"MovieLens CF test RMSE: {test_rmse:.4f} ({len(test_ratings):,} held-out ratings)")

    movie_id_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        MODEL_PATH,
        user_factors=user_factors,
        item_factors=item_factors,
        user_means=user_means,
        user_ids=user_ids,
        movie_ids=movie_ids,
        movie_id_to_tmdb_keys=np.array(list(movie_id_to_tmdb.keys())),
        movie_id_to_tmdb_values=np.array(list(movie_id_to_tmdb.values())),
    )
    print(f"Saved MovieLens CF model to {MODEL_PATH}")
    return test_rmse


class MovieLensCFRecommender:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not Path(model_path).exists():
            train()
        data = np.load(model_path)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.user_means = data["user_means"]
        self.user_ids = data["user_ids"]
        self.movie_ids = data["movie_ids"]
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}
        self.movie_id_to_tmdb = dict(
            zip(data["movie_id_to_tmdb_keys"].tolist(), data["movie_id_to_tmdb_values"].tolist())
        )
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}

    def sample_user_ids(self, n: int = 20) -> list[int]:
        return self.user_ids[:: max(1, len(self.user_ids) // n)][:n].tolist()

    def recommend_tmdb_ids(
        self, user_id: int, k: int = 10, restrict_to: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Returns [(tmdb_id, predicted_rating), ...] for the given MovieLens user,
        restricted to movies that map onto a TMDB id. If `restrict_to` is given
        (e.g. our current TMDB catalog's ids), searches the user's full ranked list
        until k matches are found within it, since coverage against any specific
        catalog subset can be sparse near the top of the ranking."""
        if user_id not in self.user_index:
            return []
        u_idx = self.user_index[user_id]
        scores = self.user_means[u_idx] + self.user_factors[u_idx] @ self.item_factors.T
        ranked_movie_idx = np.argsort(-scores)

        results = []
        for m_idx in ranked_movie_idx:
            movie_id = int(self.movie_ids[m_idx])
            tmdb_id = self.movie_id_to_tmdb.get(movie_id)
            if tmdb_id is None:
                continue
            if restrict_to is not None and tmdb_id not in restrict_to:
                continue
            results.append((tmdb_id, float(scores[m_idx])))
            if len(results) >= k:
                break
        return results


if __name__ == "__main__":
    train()
    rec = MovieLensCFRecommender()
    sample_user = rec.sample_user_ids(1)[0]
    print(f"\nRecommendations for MovieLens user {sample_user}:")
    for tmdb_id, score in rec.recommend_tmdb_ids(sample_user):
        print(f"  {score:.2f}  tmdb_id={tmdb_id}")

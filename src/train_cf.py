"""Collaborative filtering via truncated SVD (matrix factorization) over the
user-item ratings matrix, in the spirit of the Netflix Prize-era approaches.

Learns latent factors for users and movies such that (user_factors @ item_factors.T)
approximates the rating matrix, then recommends the highest-predicted unrated movies.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

from src.data import load_movielens
from src.evaluate import rmse

MODEL_PATH = Path("models/cf_model.npz")
N_FACTORS = 50


def _build_matrix(ratings: pd.DataFrame, user_ids, movie_ids):
    user_index = {u: i for i, u in enumerate(user_ids)}
    movie_index = {m: i for i, m in enumerate(movie_ids)}
    rows = ratings["userId"].map(user_index)
    cols = ratings["movieId"].map(movie_index)
    matrix = coo_matrix(
        (ratings["rating"].values, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    ).tocsr()
    return matrix, user_index, movie_index


def train(n_factors: int = N_FACTORS):
    ratings, movies = load_movielens()
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())

    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    matrix, user_index, movie_index = _build_matrix(train_ratings, user_ids, movie_ids)

    user_means = np.asarray(matrix.sum(axis=1)).flatten()
    counts = np.diff(matrix.tocsr().indptr)
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
    print(f"Test RMSE: {test_rmse:.4f} (n_factors={k}, {len(test_ratings):,} held-out ratings)")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    np.savez(
        MODEL_PATH,
        user_factors=user_factors,
        item_factors=item_factors,
        user_means=user_means,
        user_ids=user_ids,
        movie_ids=movie_ids,
    )
    print(f"Saved model to {MODEL_PATH}")
    return test_rmse


class CFRecommender:
    """Loads a trained model and serves top-N recommendations for a user."""

    def __init__(self, model_path: Path = MODEL_PATH):
        data = np.load(model_path)
        self.user_factors = data["user_factors"]
        self.item_factors = data["item_factors"]
        self.user_means = data["user_means"]
        self.user_ids = data["user_ids"]
        self.movie_ids = data["movie_ids"]
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_index = {m: i for i, m in enumerate(self.movie_ids)}

    def recommend(self, user_id: int, rated_movie_ids: set, k: int = 10):
        if user_id not in self.user_index:
            return []
        u_idx = self.user_index[user_id]
        scores = self.user_means[u_idx] + self.user_factors[u_idx] @ self.item_factors.T
        ranked = np.argsort(-scores)
        results = []
        for m_idx in ranked:
            movie_id = int(self.movie_ids[m_idx])
            if movie_id in rated_movie_ids:
                continue
            results.append((movie_id, float(scores[m_idx])))
            if len(results) >= k:
                break
        return results


if __name__ == "__main__":
    train()

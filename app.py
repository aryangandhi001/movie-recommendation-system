"""Gradio demo for the movie recommender: two tabs, two recommendation styles."""

import gradio as gr

from src.content_based import ContentRecommender
from src.data import load_movielens
from src.train_cf import CFRecommender

ratings, movies = load_movielens()
movie_id_to_title = dict(zip(movies["movieId"], movies["title"]))
content_rec = ContentRecommender()
cf_rec = CFRecommender()

KNOWN_USER_IDS = sorted(ratings["userId"].unique().tolist())


def recommend_similar(title: str):
    if not title:
        return "Pick a movie first."
    results = content_rec.recommend(title, k=10)
    if not results:
        return f"No match found for '{title}'."
    lines = [f"Because you watched **{title}**:\n"]
    for rec_title, genres, sim in results:
        lines.append(f"- **{rec_title}** ({genres}) — similarity {sim:.2f}")
    return "\n".join(lines)


def recommend_for_user(user_id: int):
    if user_id is None:
        return "Pick a user first."
    rated = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    results = cf_rec.recommend(int(user_id), rated, k=10)
    if not results:
        return f"No recommendations available for user {user_id}."
    lines = [f"Recommended for **user {user_id}** ({len(rated)} movies already rated):\n"]
    for movie_id, score in results:
        title = movie_id_to_title.get(movie_id, f"movie {movie_id}")
        lines.append(f"- **{title}** — predicted rating {score:.2f}")
    return "\n".join(lines)


with gr.Blocks(title="Movie Recommender") as demo:
    gr.Markdown(
        "# Movie Recommendation System\n"
        "Trained on the MovieLens dataset (100k ratings, 610 users, ~9,700 movies)."
    )

    with gr.Tab("Similar movies (content-based)"):
        gr.Markdown("Pick a movie you like — recommends similar movies by genre.")
        title_input = gr.Dropdown(
            choices=sorted(movies["title"].tolist()),
            label="Movie",
            filterable=True,
        )
        similar_btn = gr.Button("Find similar movies")
        similar_output = gr.Markdown()
        similar_btn.click(recommend_similar, inputs=title_input, outputs=similar_output)

    with gr.Tab("Recommended for you (collaborative filtering)"):
        gr.Markdown(
            "Pick a user ID from the dataset — recommends movies based on rating patterns "
            "of similar users."
        )
        user_input = gr.Dropdown(choices=KNOWN_USER_IDS, label="User ID")
        user_btn = gr.Button("Get recommendations")
        user_output = gr.Markdown()
        user_btn.click(recommend_for_user, inputs=user_input, outputs=user_output)

if __name__ == "__main__":
    demo.launch()

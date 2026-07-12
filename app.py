"""Gradio demo: content-based "similar titles" search, taste-profile
personalization, classic collaborative filtering, and trending rankings
over Hollywood/Bollywood movies and TV series (2005-2025)."""

import os

import gradio as gr

from src.content_based import ContentRecommender
from src.movielens_cf import MovieLensCFRecommender

rec = ContentRecommender()
cf_rec = MovieLensCFRecommender()
ALL_LABELS = sorted(rec.all_labels())
CATALOG_MOVIE_TMDB_IDS = set(rec.titles[rec.titles["media_type"] == "movie"]["tmdb_id"])
SAMPLE_USER_IDS = cf_rec.sample_user_ids(20)


def recommend_similar(label: str):
    if not label:
        return "Pick a title first."
    results = rec.recommend(label, k=10)
    if not results:
        return f"No match found for '{label}'."
    lines = [f"Because you watched **{label}**:\n"]
    for rec_label, genres, sim in results:
        lines.append(f"- **{rec_label}** ({genres}) — similarity {sim:.2f}")
    return "\n".join(lines)


def recommend_for_you(liked: list[str]):
    if not liked:
        return "Pick at least one title you like first."
    results = rec.recommend_for_profile(liked, k=10)
    if not results:
        return "No recommendations found."
    lines = [f"Because you like {', '.join(liked)}:\n"]
    for rec_label, genres, score in results:
        lines.append(f"- **{rec_label}** ({genres}) — match {score:.2f}")
    return "\n".join(lines)


def recommend_classic_cf(user_id: int):
    if user_id is None:
        return "Pick a sample MovieLens user first."
    results = cf_rec.recommend_tmdb_ids(int(user_id), k=10, restrict_to=CATALOG_MOVIE_TMDB_IDS)
    if not results:
        return f"No recommendations found for user {user_id}."
    lines = [
        f"**Classic collaborative filtering** for MovieLens user {user_id} "
        "(real historical ratings, matrix factorization):\n"
    ]
    for tmdb_id, score in results:
        label = rec.label_for_tmdb_id(tmdb_id, "movie") or f"tmdb:{tmdb_id}"
        lines.append(f"- **{label}** — predicted rating {score:.2f}")
    return "\n".join(lines)


def show_trending(media_type: str):
    mt = {"Movies": "movie", "TV Series": "tv", "Both": None}[media_type]
    df = rec.trending(k=15, media_type=mt)
    lines = ["**Top trending:**\n"]
    for _, row in df.iterrows():
        kind = "Movie" if row["media_type"] == "movie" else "TV"
        lines.append(
            f"- **{row['title']}** ({row['year']}) [{kind}] — "
            f"{row['weighted_rating']:.1f}★ ({int(row['vote_count']):,} votes) — {row['genres']}"
        )
    return "\n".join(lines)


with gr.Blocks(title="Movie & TV Recommender") as demo:
    gr.Markdown(
        "# Movie & TV Recommendation System\n"
        "Hollywood + Bollywood, movies + TV series, 2005–2025 — live data from TMDB."
    )

    with gr.Tab("Recommended for you"):
        gr.Markdown(
            "Pick a few titles you like — builds a personal taste profile and ranks "
            "the whole catalog against it (blended with a quality/trending signal)."
        )
        liked_input = gr.Dropdown(
            choices=ALL_LABELS, label="Titles you like", multiselect=True, filterable=True
        )
        profile_btn = gr.Button("Get personalized recommendations", variant="primary")
        profile_output = gr.Markdown()
        profile_btn.click(recommend_for_you, inputs=liked_input, outputs=profile_output)

    with gr.Tab("Similar titles"):
        gr.Markdown("Pick a movie or show you like — recommends similar titles by genre + plot.")
        title_input = gr.Dropdown(choices=ALL_LABELS, label="Title", filterable=True)
        similar_btn = gr.Button("Find similar titles")
        similar_output = gr.Markdown()
        similar_btn.click(recommend_similar, inputs=title_input, outputs=similar_output)

    with gr.Tab("Classic collaborative filtering"):
        gr.Markdown(
            "Demonstrates real 'users like you also liked...' collaborative filtering "
            "(SVD matrix factorization) trained on real MovieLens user ratings, mapped "
            "onto our TMDB catalog. Pick a sample historical user to see their "
            "recommendations. Limited to movies released before ~2018, since that's "
            "the real ratings data this is trained on."
        )
        user_input = gr.Dropdown(choices=SAMPLE_USER_IDS, label="Sample MovieLens user ID")
        cf_btn = gr.Button("Get recommendations")
        cf_output = gr.Markdown()
        cf_btn.click(recommend_classic_cf, inputs=user_input, outputs=cf_output)

    with gr.Tab("Trending"):
        media_filter = gr.Radio(["Both", "Movies", "TV Series"], value="Both", label="Filter")
        trending_output = gr.Markdown()
        media_filter.change(show_trending, inputs=media_filter, outputs=trending_output)
        demo.load(show_trending, inputs=media_filter, outputs=trending_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

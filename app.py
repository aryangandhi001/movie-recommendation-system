"""Gradio demo: content-based "similar titles" search + trending rankings
over Hollywood/Bollywood movies and TV series (2005-2025)."""

import gradio as gr

from src.content_based import ContentRecommender

rec = ContentRecommender()
ALL_LABELS = sorted(rec.all_labels())


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

    with gr.Tab("Similar titles"):
        gr.Markdown("Pick a movie or show you like — recommends similar titles by genre + plot.")
        title_input = gr.Dropdown(choices=ALL_LABELS, label="Title", filterable=True)
        similar_btn = gr.Button("Find similar titles")
        similar_output = gr.Markdown()
        similar_btn.click(recommend_similar, inputs=title_input, outputs=similar_output)

    with gr.Tab("Trending"):
        media_filter = gr.Radio(["Both", "Movies", "TV Series"], value="Both", label="Filter")
        trending_output = gr.Markdown()
        media_filter.change(show_trending, inputs=media_filter, outputs=trending_output)
        demo.load(show_trending, inputs=media_filter, outputs=trending_output)

if __name__ == "__main__":
    demo.launch()

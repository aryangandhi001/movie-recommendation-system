# Movie & TV Recommendation System

A content-based recommender covering **Hollywood and Bollywood movies and TV series
released 2005-2025**, built on live data from [TMDB](https://www.themoviedb.org/).

## Approach

TMDB doesn't expose individual users' rating histories, so classic collaborative
filtering ("users like you also liked...") isn't possible on this data. Instead:

- **Content-based recommendations** (`src/content_based.py`) — TF-IDF over each
  title's genres + plot overview, ranked by cosine similarity. Powers
  "because you watched X, you might like Y" for any movie or show, regardless of
  how many ratings it has (no cold-start problem).
- **Trending ranking** — an IMDB-style Bayesian weighted rating
  (`vote_average` pulled toward the global mean based on `vote_count`), so a
  5.0★ title with 3 votes doesn't outrank an 8.5★ title with 5,000 votes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Get a free TMDB API Read Access Token at https://www.themoviedb.org/settings/api,
then set it as an environment variable:

```bash
set TMDB_TOKEN=your_token_here      # Windows cmd
$env:TMDB_TOKEN="your_token_here"   # PowerShell
```

Fetch the dataset (Hollywood + Bollywood, movies + TV, 2005-2025):

```bash
python -m src.tmdb_data
```

Build the recommender index:

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
  tmdb_data.py       # fetches + caches titles from the TMDB API
  content_based.py   # TF-IDF + cosine similarity, trending ranking
app.py               # Gradio demo
```

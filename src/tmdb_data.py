"""Fetches Bollywood + Hollywood movies and TV series (2005-2025) from TMDB.

Requires a TMDB API Read Access Token, set via the TMDB_TOKEN environment
variable (get one free at https://www.themoviedb.org/settings/api).
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

API_BASE = "https://api.themoviedb.org/3"
CACHE_PATH = Path("data/raw/titles.parquet")
YEAR_START, YEAR_END = 2005, 2025
MAX_PAGES_PER_QUERY = 15  # 20 results/page -> up to 300 titles per (query, sort) combo
MIN_VOTE_COUNT_FOR_VOTE_SORT = 50  # avoid vote_count.desc surfacing obscure titles with 1-2 votes


def _headers():
    token = os.environ.get("TMDB_TOKEN")
    if not token:
        raise RuntimeError("Set the TMDB_TOKEN environment variable to your TMDB API Read Access Token.")
    return {"Authorization": f"Bearer {token}", "accept": "application/json"}


def _get_genre_map(media_type: str) -> dict[int, str]:
    resp = requests.get(f"{API_BASE}/genre/{media_type}/list", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return {g["id"]: g["name"] for g in resp.json()["genres"]}


def _discover(
    media_type: str, language: str | None, region: str | None, label: str, sort_by: str
) -> list[dict]:
    # TMDB's discover query params and response fields use different names for movies.
    query_date_field = "primary_release_date" if media_type == "movie" else "first_air_date"
    response_date_field = "release_date" if media_type == "movie" else "first_air_date"
    genre_map = _get_genre_map(media_type)
    results = []

    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        params = {
            f"{query_date_field}.gte": f"{YEAR_START}-01-01",
            f"{query_date_field}.lte": f"{YEAR_END}-12-31",
            "sort_by": sort_by,
            "page": page,
            "include_adult": "false",
        }
        if sort_by == "vote_count.desc":
            params["vote_count.gte"] = MIN_VOTE_COUNT_FOR_VOTE_SORT
        if language:
            params["with_original_language"] = language
        if region:
            params["region"] = region

        resp = requests.get(f"{API_BASE}/discover/{media_type}", headers=_headers(), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data["results"]:
            title = item.get("title") or item.get("name")
            date = item.get(response_date_field) or ""
            results.append({
                "tmdb_id": item["id"],
                "media_type": media_type,
                "title": title,
                "year": int(date[:4]) if date[:4].isdigit() else None,
                "genres": "|".join(genre_map.get(gid, "") for gid in item.get("genre_ids", [])),
                "overview": item.get("overview", ""),
                "popularity": item.get("popularity", 0.0),
                "vote_average": item.get("vote_average", 0.0),
                "vote_count": item.get("vote_count", 0),
                "original_language": item.get("original_language", ""),
                "poster_path": item.get("poster_path", ""),
                "source_query": label,
            })

        if page >= data.get("total_pages", 1):
            break
        time.sleep(0.05)  # stay well under TMDB's rate limit

    print(f"  {label} ({sort_by}): {len(results)} titles")
    return results


def fetch_all() -> pd.DataFrame:
    queries = [
        ("movie", "en", "US", "hollywood_movies"),
        ("movie", "hi", "IN", "bollywood_movies"),
        ("tv", "en", "US", "hollywood_series"),
        ("tv", "hi", "IN", "bollywood_series"),
    ]
    all_results = []
    for media_type, language, region, label in queries:
        # popularity.desc surfaces what's trending *right now*; vote_count.desc
        # surfaces titles with lasting/widespread viewership even if not currently
        # trending (e.g. older hits) -- without this, well-known-but-not-currently-
        # trending titles fall outside the page cutoff entirely.
        all_results.extend(_discover(media_type, language, region, label, "popularity.desc"))
        all_results.extend(_discover(media_type, language, region, label, "vote_count.desc"))

    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=["tmdb_id", "media_type"]).reset_index(drop=True)
    return df


def load_titles(force_refresh: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force_refresh:
        return pd.read_parquet(CACHE_PATH)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_all()
    df.to_parquet(CACHE_PATH)
    print(f"Cached {len(df):,} titles to {CACHE_PATH}")
    return df


if __name__ == "__main__":
    df = load_titles(force_refresh=True)
    print(df["source_query"].value_counts())
    print(df.head())

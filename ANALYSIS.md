# Movie & TV Recommendation System — Full Technical Report

A complete, exhaustive walkthrough: what this project does, why every
design decision was made, every function in the codebase, every bug hit
and how it was diagnosed and fixed, how it's deployed, and what's honestly
still missing.

---

## 1. What this project is, and why it changed shape twice

This started as a classic MovieLens-based recommender (SVD collaborative
filtering + content-based filtering), the standard academic benchmark for
recommender systems. It was then **rebuilt from scratch** on request to
cover something genuinely more current and personally relevant: Hollywood
*and* Bollywood, movies *and* TV series, released 2005–2025 — MovieLens
alone is English-language-only with ratings capped at 2018, which doesn't
reflect either "currently relevant" or "culturally broad" content.

That rebuild introduced a real constraint worth stating plainly: **TMDB
(the live data source used for the rebuild) doesn't expose individual
users' rating histories** the way MovieLens does. So true collaborative
filtering ("users like you also liked...") isn't directly possible on the
TMDB catalog. The system that resulted is a **hybrid**, built to work
around that constraint honestly rather than fake it:

- **Content-based filtering** and a **taste-profile personalization layer**
  run on the live TMDB catalog (2005–2025, Hollywood + Bollywood).
- **Real collaborative filtering** (SVD matrix factorization) still runs,
  but on the original MovieLens ratings data specifically — kept as a
  separate, clearly-labeled tab, since it's real historical user behavior
  data that TMDB simply doesn't have. It's bridged onto the TMDB catalog
  via MovieLens' own `links.csv` (movieId → tmdbId mapping), so the two
  data sources talk to the same catalog rather than existing as two
  disconnected demos.

This is the honest architecture: two genuinely different data sources,
each used for what it's actually good for, not one system pretending to
be more unified than it is.

---

## 2. Architecture overview

```
src/
  tmdb_data.py       -- fetches + caches the live TMDB catalog
  content_based.py   -- TF-IDF similarity, trending ranking, taste-profile personalization
  movielens_cf.py    -- SVD collaborative filtering on real MovieLens ratings
app.py               -- Gradio demo, 4 tabs
```

Four user-facing capabilities, each backed by a different piece of the
pipeline:

1. **Recommended for you** (personalization) — pick a few titles you like,
   get a personalized ranking of the whole catalog.
2. **Similar titles** (content-based) — pick one title, get titles similar
   by genre + plot.
3. **Classic collaborative filtering** — pick a real historical MovieLens
   user, see what the SVD model predicts they'd like, mapped onto the
   modern catalog.
4. **Trending** — a quality-adjusted popularity ranking.

---

## 3. File-by-file, function-by-function walkthrough

### `src/tmdb_data.py`

```python
def _discover(media_type, language, region, label, sort_by) -> list[dict]:
```
The core fetcher. Calls TMDB's `/discover/{movie|tv}` endpoint, paginated
(`MAX_PAGES_PER_QUERY = 15` pages × 20 results/page = up to 300 titles per
call), filtered to the 2005–2025 release window, and optionally filtered
by `language`/`region` (used to separate Hollywood, `en`/`US`, from
Bollywood, `hi`/`IN`). Genre IDs returned by TMDB are numeric; `_get_genre_map`
fetches the id→name mapping once per media type so genres can be stored as
readable strings (`"Action|Adventure|Comedy"`) rather than opaque IDs.

```python
def fetch_all() -> pd.DataFrame:
    queries = [
        ("movie", "en", "US", "hollywood_movies"),
        ("movie", "hi", "IN", "bollywood_movies"),
        ("tv", "en", "US", "hollywood_series"),
        ("tv", "hi", "IN", "bollywood_series"),
    ]
    for media_type, language, region, label in queries:
        all_results.extend(_discover(..., "popularity.desc"))
        all_results.extend(_discover(..., "vote_count.desc"))
```
Runs 4 category queries (Hollywood movies, Bollywood movies, Hollywood
series, Bollywood series), and — this is the fix for a real bug covered
below — runs **each category twice**, once sorted by `popularity.desc` and
once by `vote_count.desc`, merging and deduplicating the results. Also
caches the whole thing to `data/raw/titles.parquet` (`load_titles`) so
repeat runs and the deployed demo don't need to re-hit the TMDB API.

### `src/content_based.py`

```python
def _weighted_rating(df, m=MIN_VOTE_COUNT) -> pd.Series:
    C = df["vote_average"].mean()
    v = df["vote_count"]
    R = df["vote_average"]
    return (v / (v + m)) * R + (m / (v + m)) * C
```
The "Trending" ranking isn't a raw sort by rating — that would let a title
with a perfect 10.0 from 3 votes outrank a title with an 8.5 from 5,000
votes, which is obviously wrong. This is the standard IMDB-style Bayesian
weighted-rating formula: it pulls a title's score toward the *global*
average rating `C`, proportionally to how few votes it has (`m` is a
tunable "how many votes before we start trusting the raw average"
threshold). A title with very few votes ends up close to the global
average; a title with thousands of votes ends up close to its own raw
average, since `v/(v+m)` approaches 1 as `v` grows large relative to `m`.

```python
def build():
    text = titles["genres"] (pipe-replaced with spaces) + " " + titles["overview"]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20_000)
    vectors = vectorizer.fit_transform(text)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(vectors)
```
Builds the content-similarity index: each title's genres and plot overview
are concatenated into one text blob, TF-IDF vectorized, and indexed with
scikit-learn's `NearestNeighbors` using cosine distance (`algorithm="brute"`
is appropriate here — at ~1,300 titles, brute-force cosine search is fast
enough that an approximate-nearest-neighbor index would be unnecessary
complexity). The fitted vectorizer, the NN index, the raw vectors, and the
titles DataFrame are all pickled together into one `.joblib` file, so
loading the model at serving time is a single `joblib.load` — no need to
recompute anything.

```python
def recommend(self, label: str, k: int = 10):
```
"Similar titles": looks up a title's row index, queries the `NearestNeighbors`
index for its `k+1` nearest neighbors (the `+1` accounts for the title
always being its own nearest neighbor, which gets filtered out), and
returns the rest ranked by similarity.

```python
def recommend_for_profile(self, liked_labels: list[str], k=10, content_weight=0.7):
    profile_vector = np.asarray(self.vectors[liked_indices].mean(axis=0))
    profile_vector = l2_normalize(profile_vector)
    content_scores = np.asarray(self.vectors @ profile_vector.T).flatten()
    final_scores = content_weight * content_scores + (1 - content_weight) * self._trending_norm
```
"Recommended for you": this is the real personalization layer, and it's
the answer to the specific constraint that TMDB has no per-user history.
Given a handful of titles the user says they like, it builds a single
"taste profile" vector as the **mean of their TF-IDF vectors**, L2-normalizes
it (so it's comparable to the L2-normalized rows of the vector index via a
plain dot product — cosine similarity between L2-normalized vectors is
just their dot product), then scores every title in the catalog against
that profile. The final score is a **weighted blend** (`content_weight=0.7`)
of raw content similarity and the normalized trending score — this is
deliberate: pure content similarity alone tends to recommend obscure titles
that happen to share genre tags, while blending in a quality/popularity
signal keeps recommendations both personalized *and* reasonably well-regarded,
which is closer to how real recommendation systems actually behave (Netflix
doesn't only optimize for topical similarity either).

### `src/movielens_cf.py`

```python
def train(n_factors=50):
    matrix = coo_matrix((ratings, (user_idx, movie_idx))).tocsr()
    user_means = matrix.sum(axis=1) / counts
    matrix_centered = matrix - user_means (per row)
    U, sigma, Vt = svds(matrix_centered, k=n_factors)
    user_factors = U @ diag(sigma)
    item_factors = Vt.T
```
Classic SVD-based collaborative filtering (matrix factorization), applied
to the real MovieLens `ratings.csv`. The rating matrix is mean-centered per
user first (subtracting each user's average rating before factorizing) — a
standard technique that stops the factorization from being dominated by
which users simply rate everything higher or lower on average, letting the
learned latent factors capture actual *preference* rather than rating-scale
bias. `scipy.sparse.linalg.svds` computes a truncated (rank-50) SVD
directly on the sparse matrix, which is far cheaper than a dense SVD given
how sparse a real ratings matrix is (most users haven't rated most movies).
A prediction for a given (user, movie) pair is then
`user_mean + user_factors[user] @ item_factors[movie]`.

```python
movie_id_to_tmdb = dict(zip(links["movieId"], links["tmdbId"]))
```
This is the bridge between the two data sources: MovieLens ships its own
`links.csv` mapping its internal `movieId` to the corresponding TMDB ID, so
predictions computed against MovieLens's rating matrix can be translated
directly into "which TMDB catalog entry does this correspond to" — letting
the classic-CF tab recommend titles that also exist in the modern TMDB
catalog, not just MovieLens's own closed universe.

```python
def recommend_tmdb_ids(self, user_id, k=10, restrict_to=None):
```
Ranks all movies by predicted rating for a given user, walks down that
ranked list translating each to a TMDB ID via the links mapping, and — if
`restrict_to` is given (a set of TMDB IDs, i.e. "only ones in our current
catalog") — keeps searching past the top of the list until `k` matches are
found, rather than stopping early. This detail matters and is covered in
the debugging section below.

### `app.py`

Wires all of the above into four Gradio tabs. The one non-obvious detail:
`label_for_tmdb_id(tmdb_id, media_type)` and every other TMDB-ID-based
lookup in this app is always keyed on the **pair** `(tmdb_id, media_type)`,
never `tmdb_id` alone — also covered below.

---

## 4. The real debugging journey

### Bug: TMDB's "popularity" sort was silently excluding well-known older titles

Early testing surfaced that well-known titles — *Tamasha*, *Student of the
Year* — were completely absent from the fetched catalog. Investigating
directly against the TMDB API: **Tamasha's TMDB "popularity" score was
0.73; Student of the Year's was 1.17** — for comparison, a currently-trending
title's popularity score is often in the hundreds. TMDB's `popularity`
metric reflects *current* buzz/activity (recent views, searches), not
overall fame or importance — so a top-500-by-popularity cutoff was
systematically crowding out older-but-genuinely-well-known titles in favor
of whatever's algorithmically trending on TMDB *right now*.

**Fix:** fetch each category by *both* `popularity.desc` and
`vote_count.desc`, merging and deduplicating. `vote_count` accumulates over
a title's whole lifetime rather than reflecting only current buzz, so it
surfaces "lastingly well-known" titles that a pure-popularity sort misses
entirely. Verified this fixed the specific reported cases directly.

### Bug: movie release years showing as `nan`

After the initial TMDB integration, every **movie's** year field was `nan`
— but TV series years were fine. Root cause, found by inspecting the raw
API response directly: TMDB's `/discover` endpoint uses different field
names for the *query parameter* vs. the *response object*, and only for
movies. The query parameter for filtering by date is `primary_release_date`,
but the actual field in each returned movie object is `release_date` — the
code was querying with the correct parameter name but then trying to read
`primary_release_date` back out of the response objects, which doesn't
exist there, so the extracted date was always empty. (TV's parameter and
response field happen to have the *same* name, `first_air_date`, which is
why only movies were affected — an easy trap, since testing just the TV
path would have looked completely fine.) Fixed by using separate constants
for the query field name vs. the response field name.

### Bug: TMDB movie IDs and TV IDs are separate namespaces that can collide

While wiring up the classic-CF tab's MovieLens→TMDB bridge, some title
lookups were silently returning the wrong title (or `None`). Root cause:
**TMDB's movie IDs and TV IDs are independent numbering spaces** — a
movie with ID `1891` and a completely unrelated TV show with ID `1891`
can both exist, and nothing about the bare integer distinguishes them. Any
code that looked up "the title with this `tmdb_id`" without *also*
checking `media_type` could silently return the wrong entry, or fail to
find an entry that actually existed under the other media type. **Fix:**
every catalog lookup in this project is keyed on the pair
`(tmdb_id, media_type)`, never `tmdb_id` alone.

### Bug: classic-CF recommendations came back nearly empty against the modern catalog

Once the MovieLens→TMDB bridge was working, restricting recommendations to
"only titles that exist in our current 2005–2025 catalog" initially
returned very few results per user — sometimes 0–2 out of a requested 10.
Root cause: MovieLens's own top-ranked recommendations for a given user
are frequently *older classics* (pre-2005, since MovieLens itself skews
toward well-established films), which simply aren't in this project's
2005–2025-only catalog at all. The original implementation capped its
search at the top `k` overall-ranked movies before checking catalog
membership, so if none of those top-k happened to be post-2005, the
restricted result set came back nearly empty even though *plenty* of
catalog-eligible recommendations existed further down the user's full
ranked list. **Fix:** `recommend_tmdb_ids` now keeps walking down the
*entire* ranked list (not just the top-k) until `k` catalog-eligible
matches are actually found.

### Bug: local pip install silently missing packages

Twice during setup, a background `pip install -r requirements.txt` run
reported success (exit code 0) but several packages — `pandas`/`gradio`
the first time, `pyarrow`/`joblib` the second — were simply absent
afterward. Diagnosed by explicitly checking `pip show <package>` /
`pip list` against the actual venv rather than trusting the install
command's exit code. Root cause was never fully pinned down (possibly a
background-process buffering/truncation issue specific to this
environment), but the practical lesson was: **verify a dependency install
actually landed before trusting it**, rather than assuming success from a
clean exit code.

---

## 5. Results / what the numbers actually mean

There's no single "accuracy" number for a recommender system the way
there is for a classifier — the honest way to characterize this system's
quality:

- **Content-based similarity** is directly inspectable: querying "similar
  titles" for a known film reliably returns genuinely genre/theme-adjacent
  titles (verified by spot-checking specific queries during development).
- **Classic CF's real, measurable number**: RMSE ≈ 0.93 on held-out
  MovieLens ratings (a 0.5–5 star scale) — in line with published baseline
  results for SVD-based collaborative filtering on this dataset.
- **Trending ranking** is a deterministic, explainable formula (the
  Bayesian weighted rating above), not a learned model — its "correctness"
  is that it doesn't let low-vote-count titles game the top of the list,
  which is directly verifiable by inspection.

---

## 6. Honest limitations and what's actually missing

- **No true collaborative filtering on the live 2019–2025 catalog.** This
  is a fundamental data-availability constraint, not a shortcut — TMDB
  simply doesn't expose per-user rating histories, and there's no way to
  fake genuine collaborative signal on data that doesn't have per-user
  structure. The classic-CF tab is real, but it's necessarily scoped to
  pre-2018 MovieLens data.
- **No offline evaluation of the personalization layer.** The taste-profile
  blend (`content_weight=0.7`) was chosen by judgment/inspection, not by a
  measured metric — there's no held-out "does this actually predict what a
  real user with this taste profile would rate highly" test.
- **The `content_weight=0.7` blend ratio is a single global constant,**
  not tuned per user or validated against any ground truth.

---

## 7. Interview-ready summary

*"This is a hybrid movie/TV recommender — content-based filtering and a
taste-profile personalization layer on a live, current (2005–2025,
Hollywood + Bollywood) TMDB catalog, plus real SVD-based collaborative
filtering on the original MovieLens dataset, bridged onto the same catalog
via MovieLens's own ID-mapping table. The honest architectural decision was
recognizing that TMDB doesn't expose per-user rating histories, so true
collaborative filtering isn't fakeable on that data — instead of
pretending otherwise, I kept classic CF on the data source that actually
has it. Along the way I hit and fixed several real bugs: TMDB's popularity
metric was silently excluding older well-known titles in favor of
whatever's trending right now, a query-vs-response field name mismatch was
zeroing out every movie's release year, and TMDB's movie/TV ID namespaces
can collide on the same integer, which needed every lookup in the system
to key on (ID, media_type) pairs instead of ID alone."*

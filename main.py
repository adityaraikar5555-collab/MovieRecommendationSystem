import os
import pickle
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

# Shared httpx client (connection pooling — big performance win)
_http_client: Optional[httpx.AsyncClient] = None


# =========================
# LIFESPAN (replaces deprecated @app.on_event)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX, _http_client

    # Startup
    _http_client = httpx.AsyncClient(timeout=20, limits=httpx.Limits(max_connections=20))

    for name, path in [
        ("df", os.path.join(BASE_DIR, "df.pkl")),
        ("indices", os.path.join(BASE_DIR, "indices.pkl")),
        ("tfidf_matrix", os.path.join(BASE_DIR, "tfidf_matrix.pkl")),
        ("tfidf", os.path.join(BASE_DIR, "tfidf.pkl")),
    ]:
        if not os.path.exists(path):
            raise RuntimeError(f"Missing pickle file: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if name == "df":
            df = obj
        elif name == "indices":
            indices_obj = obj
        elif name == "tfidf_matrix":
            tfidf_matrix = obj
        elif name == "tfidf":
            tfidf_obj = obj

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    TITLE_TO_IDX = _build_title_to_idx_map(indices_obj)
    print(f"✅ Loaded {len(df)} movies | TF-IDF matrix: {tfidf_matrix.shape}")

    yield  # app runs here

    # Shutdown
    await _http_client.aclose()


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Uses shared connection pool for all TMDB requests."""
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        r = await _http_client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"TMDB request error: {repr(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}: {r.text[:200]}")

    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[TMDBMovieCard]:
    return [
        TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or m.get("name") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
        for m in (results or [])[:limit]
    ]


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF HELPERS
# =========================
def _build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError("indices.pkl must be dict or pandas Series-like")


def _get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(status_code=404, detail=f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = _get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


async def _attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """Searches TMDB for a local title to fetch its poster. Never crashes."""
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"status": "ok", "message": "Movie Recommender API v4", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "movies_loaded": len(df) if df is not None else 0,
        "tfidf_ready": tfidf_matrix is not None,
    }


# ---------- HOME FEED ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
        elif category in {"popular", "top_rated", "upcoming", "now_playing"}:
            data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        else:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        return await tmdb_cards_from_results(data.get("results", []), limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF + Genre (PARALLEL REQUESTS) ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(status_code=404, detail=f"No TMDB movie found for: {query}")

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # --- TF-IDF recs ---
    recs: List[Tuple[str, float]] = []
    try:
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    # KEY PERF FIX: Fetch all TF-IDF posters IN PARALLEL instead of one-by-one
    tfidf_cards = await asyncio.gather(
        *[_attach_tmdb_card_by_title(title) for title, _ in recs],
        return_exceptions=False,
    )
    tfidf_items = [
        TFIDFRecItem(title=title, score=score, tmdb=card)
        for (title, score), card in zip(recs, tfidf_cards)
    ]

    # --- Genre recs ---
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {"with_genres": genre_id, "language": "en-US", "sort_by": "popularity.desc", "page": 1},
        )
        cards = await tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )
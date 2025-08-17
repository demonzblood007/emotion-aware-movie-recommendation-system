## Emotion-Aware Movie Recommendation System

An end-to-end movie recommendation project that blends classic user–item signals with LLM-powered understanding of user taste and emotion. It ingests MovieLens data, builds user taste profiles enriched via web search, and exposes recommendations through a FastAPI backend and Streamlit frontend.

### Key Features
- **LLM Taste Profiles (LangChain + Gemini + Tavily):** Generates narrative user profiles from top-liked/disliked movies using `Gemini 2.0 Flash` and Tavily search enrichment.
- **Emotion Signals (BERT):** A local BERT model (in `bert_emotion_model/`) to incorporate emotion-aware insights in recommendations and summaries.
- **Database-backed Profiles (PostgreSQL + SQLAlchemy):** Stores per-user taste profiles for reuse and auditing.
- **Serving:** FastAPI service (`api.py`) and a Streamlit app (`frontend/streamlit_app.py`).

---

## Project Structure
```
.
├─ api.py                        # FastAPI app (recommendations / profile endpoints)
├─ chatbot.py                    # Chat-based interactions (optional)
├─ final_recommender.py          # Core recommendation logic
├─ frontend/
│  └─ streamlit_app.py          # Simple UI for recommendations
├─ schema/
│  ├─ test_db.py                # Profile generator (Gemini + Tavily) for first N users
│  ├─ user_model.py             # SQLAlchemy models & DB engine
│  ├─ summary_generator.py      # (Aux) LLM summary utilities
│  └─ webscraper.py             # (Aux) scraping utilities
├─ database_movielens/
│  ├─ filtered_data.csv         # ratings subset
│  ├─ movies.csv                # movieId → title, genres
│  └─ links.csv                 # movieId → imdbId / tmdbId
├─ bert_emotion_model/          # Local BERT weights and tokenizer
├─ pyproject.toml               # Dependencies (managed via uv or pip)
└─ README.md
```

---

## Prerequisites
- Python 3.12+
- PostgreSQL running locally with a database (default in code: `movierec`)
- API keys
  - `GOOGLE_API_KEY` (for Gemini)
  - `TAVILY_API_KEY` (for Tavily Search)

Create a `.env` in the repo root:
```
GOOGLE_API_KEY="<your_google_api_key>"
TAVILY_API_KEY="<your_tavily_api_key>"
```

Note: `schema/user_model.py` currently uses a hard-coded Postgres URL. Update it to match your local DB or switch to env-based configuration (see Roadmap).

---

## Setup
Using uv (recommended):
```
# Install uv if needed: https://docs.astral.sh/uv/
uv sync
```

Using pip:
```
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

Ensure Postgres is up and the database exists. The default connection in `schema/user_model.py` is:
```
postgresql://postgres:Bhavik30@localhost:5432/movierec
```
Change this to your credentials before running in production.

---

## Data
Place the MovieLens CSVs under `database_movielens/`:
- `filtered_data.csv` (userId, movieId, rating)
- `movies.csv` (movieId, title, genres)
- `links.csv` (movieId, imdbId, tmdbId)

---

## Generate User Profiles (LLM + Search)
The pipeline enriches top liked/disliked movies via Tavily, then asks Gemini to summarize the user’s taste.

Run (generates for first 5 users by default):
```
python schema/test_db.py
```
What happens:
- Picks each user’s top 3 liked (rating ≥ 4) and top 3 disliked (rating ≤ 2)
- For each title, uses Tavily to fetch brief context (director, cast, plot)
- Calls Gemini to synthesize a concise profile summary
- Upserts into `user_profiles` (PostgreSQL)

Reset and regenerate (only first 5 users):
- Quick reset (truncate):
```
TRUNCATE TABLE user_profiles RESTART IDENTITY;
```
- Or drop/recreate table (Python):
```python
from schema.user_model import Base, engine, UserProfile
Base.metadata.drop_all(engine, tables=[UserProfile.__table__], checkfirst=True)
Base.metadata.create_all(engine, tables=[UserProfile.__table__])
```
Then rerun `python schema/test_db.py`.

---

## Run the Services
- API (dev):
```
uvicorn api:app --reload
```
- Streamlit UI:
```
streamlit run frontend/streamlit_app.py
```

---

## Configuration & Environment
- `.env`: `GOOGLE_API_KEY`, `TAVILY_API_KEY`
- Postgres URL: currently hard-coded in `schema/user_model.py`; update it or refactor to use env vars (see Roadmap)

---

## Notes on Search Credits and Optimization
- Tavily free tier typically offers around 1,000 credits/month; each search ≈ 1 credit.
- Current usage: up to 6 searches/user (3 liked + 3 disliked titles).
- Optimizations in place:
  - In-memory caching per run to avoid repeated lookups.
- Recommended next optimizations:
  - Persistent cache (disk/DB) for movie metadata across runs
  - Limit searches to most informative titles (e.g., diversity by genre/director)
  - Retry with backoff and graceful degradation if search fails

---

## How It Works (High Level)
1. Read user ratings and join with movie metadata (title, genres, ids)
2. Select strong-signal titles (top/bottom by rating)
3. Enrich each title via Tavily (director, cast, plot)
4. Prompt Gemini with structured context → narrative user profile
5. Store profile and aggregates in Postgres for downstream use

---

## Roadmap
- Replace hard-coded DB URL with env-driven configuration (e.g., `DATABASE_URL`)
- Migrate LangChain orchestration to LangGraph for better reliability and retries
- Persistent metadata cache (TMDb/IMDb API integration as an alternative to generic search)
- Integrate BERT emotion signals directly into profile prompts and ranking logic
- Batch profile generation with rate limiting and resumable checkpoints
- Evaluation harness (offline metrics + prompt evals)
- Dockerfile and docker-compose (API + DB + Streamlit)
- CI/CD with tests and pre-commit (formatting, linting)
- Fine-tuned small model for summaries to reduce latency/cost

---

## Troubleshooting
- Import errors for Tavily:
  - Use `from langchain_tavily import TavilySearch` and ensure `langchain-tavily` is installed
- Duplicate key error on reruns:
  - Either truncate table or rely on the script’s upsert logic (now updates-existing)
- Missing `psycopg2`:
  - Ensure `psycopg2-binary` is installed (included in `pyproject.toml`)

---

## License
Add your preferred license (MIT/Apache-2.0/etc.).

## Acknowledgements
- MovieLens dataset
- LangChain, Google Gemini, Tavily Search

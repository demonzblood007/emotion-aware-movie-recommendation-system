# Emotion-Aware Movie Recommendation System

A research-driven recommendation system that addresses a fundamental gap in traditional approaches: **the role of emotions in content selection**. Unlike collaborative filtering or content-based methods that rely solely on historical ratings and metadata, this system incorporates real-time emotional context to deliver contextually relevant recommendations.

## Problem Statement

Traditional recommendation systems (Netflix, Amazon, etc.) operate on two primary paradigms:
- **Collaborative Filtering**: "Users similar to you liked this"
- **Content-Based Filtering**: "You liked movies with these features, here are similar ones"

Both approaches overlook a critical human factor: **emotions drive media consumption decisions**. Whether seeking mood congruence (reinforcing current feelings) or mood management (shifting emotional state), users' emotional needs significantly influence what they want to watch. A person might choose an uplifting comedy after a stressful day or a contemplative drama when feeling reflective—decisions that cannot be captured by rating patterns alone.

## Our Approach

We designed a **hybrid architecture** that combines:
1. **Neural Collaborative Filtering (NCF)** for learning user-item preferences from historical data
2. **BERT-based emotion detection** for extracting emotional signals from natural language input
3. **Emotion-to-genre mapping** that translates detected emotions into genre preferences with configurable weighting

### Technical Architecture

**Emotion Detection Pipeline:**
- User provides natural language mood description (e.g., "feeling stressed after a long week")
- Text is processed through `SamLowe/roberta-base-go_emotions` (BERT fine-tuned on GoEmotions dataset)
- Model outputs multi-label emotion probabilities (joy, sadness, excitement, etc.)

**Emotion-to-Genre Translation:**
- Each emotion maps to **resonant genres** (reinforce the emotion) and **contrast genres** (shift the emotion)
- Example: "joy" → resonant: [Comedy, Family, Animation], contrast: [Horror, War, Thriller]
- User selects strategy: `match` (70% resonant, 30% contrast), `shift` (30% resonant, 70% contrast), or `neutral` (50/50)

**Neural Collaborative Filtering:**
- Trained on MovieLens 32M dataset
- Architecture: user embeddings + movie embeddings + genre feature vector → MLP → predicted rating
- At inference: emotion-derived genre weights are integrated as soft constraints on the NCF scoring function

**Why This Approach:**
- **NCF** captures long-term user preferences and item characteristics from large-scale data
- **Emotion signals** provide real-time context that adapts to current user state
- **Hybrid fusion** balances historical patterns with situational needs

## Unique Contributions

1. **Emotion-Aware Genre Weighting**: Dynamic genre preference vectors derived from detected emotions, not static user profiles
2. **Mood Management Strategy**: Explicit user control over whether recommendations should match or shift their current emotional state
3. **Multi-Label Emotion Detection**: Leverages fine-tuned BERT to capture nuanced emotional states (28 emotion categories from GoEmotions)
4. **Hybrid Recommendation Fusion**: Seamlessly integrates emotion-derived constraints with collaborative filtering scores

## Implementation Details

### Core Components

- **`final_recommender.py`**: PyTorch-based NCF model, emotion detection pipeline, and recommendation generation logic
- **`backend/`**: FastAPI service layer with modular routers for emotions, genres, ranking, and recommendations
- **`backend/services/`**: Service-oriented architecture separating emotion detection, genre mapping, NCF inference, and metadata enrichment

### Model Training

The NCF model (`best_model.pt`) was trained on MovieLens filtered data with:
- User and movie embeddings (32-dimensional)
- Genre multi-hot encoding (20+ genres)
- MLP layers with dropout regularization
- Trained to predict user-movie ratings

### Emotion Model

Uses Hugging Face Transformers with `SamLowe/roberta-base-go_emotions`:
- Pre-trained RoBERTa base fine-tuned on GoEmotions dataset
- Multi-label classification (28 emotion categories)
- Downloads model weights on first run (no local storage required)

## Project Structure

```
.
├─ api.py                        # FastAPI entry point
├─ backend/                      # Service layer
│  ├─ main.py                   # FastAPI app with CORS and routing
│  ├─ config.py                  # Environment-based settings
│  ├─ routers/                   # API endpoints (emotions, genres, recommendations)
│  ├─ services/                  # Business logic (emotion, genre, NCF, metadata services)
│  └─ schemas/                   # Pydantic request/response models
├─ final_recommender.py          # Core recommendation engine (PyTorch + Transformers)
├─ database_movielens/           # MovieLens dataset (filtered ratings, movies, links)
├─ best_model.pt                 # Pre-trained NCF checkpoint
├─ frontend-showcase/            # Next.js demo UI (optional)
└─ pyproject.toml                # Python dependencies
```

## Setup & Usage

### Prerequisites
- Python 3.12+
- PyTorch-compatible environment

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

### Environment Variables

Create a `.env` file:
```env
MOVIE_MODEL_PATH=./best_model.pt
EMOTION_MODEL_PATH=SamLowe/roberta-base-go_emotions
MOVIELENS_FILTERED_CSV=./database_movielens/filtered_data.csv
MOVIES_CSV=./database_movielens/movies.csv
TMDB_API_KEY=<optional>  # For movie posters and metadata
OMDB_API_KEY=<optional>  # For IMDB ratings
```

### Running the API

```bash
uvicorn api:app --reload
```

API documentation available at `http://127.0.0.1:8000/docs`

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 42,
    "mood_text": "Feeling stressed and need something uplifting",
    "strategy": "shift",
    "top_n": 10
  }'
```

Response includes:
- Detected emotions with confidence scores
- Genre weights derived from emotions
- Ranked movie recommendations with predicted ratings
- Enriched metadata (posters, IMDB/TMDB links, watch providers) if API keys are configured

## Technical Skills Demonstrated

- **Deep Learning**: Neural Collaborative Filtering, PyTorch model training and inference
- **NLP**: BERT-based emotion classification, multi-label text classification
- **ML Engineering**: Hybrid recommendation systems, feature engineering, model fusion
- **Software Architecture**: Service-oriented design, RESTful APIs, type-safe schemas
- **Data Engineering**: Large-scale dataset processing (MovieLens 32M), efficient data pipelines

## Research Context

This project was developed as part of a Bachelor's Thesis (BTP) at The LNM Institute of Information Technology, Jaipur, exploring the intersection of affective computing and recommendation systems. The work demonstrates how emotion-aware AI can create more human-centric technology that adapts to users' real-time needs rather than relying solely on historical patterns.

## Data Sources

- **MovieLens 32M Dataset**: User ratings, movie metadata, genre classifications
- **GoEmotions Dataset**: Multi-label emotion annotations for BERT fine-tuning
- **TMDB/OMDB APIs**: Movie posters, ratings, and streaming provider information

## Future Enhancements

- Personalized emotion-to-genre mappings learned from user feedback
- Temporal emotion tracking to capture mood trends over time
- Group emotion aggregation for multi-user recommendations
- Explainable recommendations that justify suggestions using both collaborative signals and emotional context

## License

This project is part of academic research. Please cite appropriately if used for research purposes.

## Acknowledgments

- MovieLens dataset (GroupLens Research, University of Minnesota)
- GoEmotions dataset and `SamLowe/roberta-base-go_emotions` model
- TMDB and OMDB APIs for metadata enrichment

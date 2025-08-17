from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import os
import uvicorn
from chatbot import run_chat, reset_user_history, count_conversations
from langgrapgh_flow import run_reco_flow
import pandas as pd
from functools import lru_cache

load_dotenv()

app = FastAPI()

class Message(BaseModel):
    type: str  # 'system', 'human', or 'ai'
    content: str

class ChatRequest(BaseModel):
    user_id: str
    user_input: str
    # chat_history is no longer required from the client, managed server-side

class ChatResponse(BaseModel):
    chat_history: List[Message]

class ResetRequest(BaseModel):
    user_id: str

class MVPRecommendRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    mood_text: Optional[str] = Field(None, description="Free-text description of current mood or desired watch vibe")
    preference: Optional[str] = Field(None, description="One of: match, shift, neutral")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations to return")

class MVPNeedInputResponse(BaseModel):
    status: str
    next_question: str
    required_field: str

class MovieItem(BaseModel):
    movieId: int
    title: str
    genres: List[str]
    avg_rating: float
    num_ratings: int

class MVPRecommendResponse(BaseModel):
    status: str
    recommended_genres: List[str]
    recommendations: List[MovieItem]

@app.post('/chat', response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # If user_input is empty, just return the current history (for initial message)
    if not req.user_input.strip():
        from chatbot import get_user_history
        history = get_user_history(req.user_id)
        out_history = [Message(**m) for m in history]
        return ChatResponse(chat_history=out_history)
    
    # Run chat logic, chatbot.py manages per-user history
    out_history_dicts = run_chat(req.user_id, req.user_input)
    out_history = [Message(**m) for m in out_history_dicts]
    return ChatResponse(chat_history=out_history)

@app.post('/reset')
def reset_endpoint(req: ResetRequest):
    reset_user_history(req.user_id)
    # Get fresh history with new initial message
    from chatbot import get_user_history, generate_initial_response, save_user_history
    
    # Get fresh history (only system message)
    new_history = get_user_history(req.user_id)
    
    # Generate new initial AI message
    initial_response = generate_initial_response()
    new_history.append({'type': 'ai', 'content': initial_response})
    
    # Save the updated history
    save_user_history(req.user_id, new_history)
    
    out_history = [Message(**m) for m in new_history]
    return {"status": "reset", "user_id": req.user_id, "chat_history": out_history}

@app.post('/langgraph-flow')
def langgraph_flow_endpoint(req: ResetRequest):
    """Trigger LangGraph flow for movie recommendations"""
    try:
        result = run_reco_flow(req.user_id)
        if not result.get("recommended_genres"):
            # Return a specific message if no genres were recommended
            return {
                "status": "success",
                "message": "We've analyzed your conversation, but couldn't determine a specific genre. Let's talk more!",
                "user_id": req.user_id,
                "genres": []
            }
            
        return {
            "status": "success",
            "message": "LangGraph flow triggered successfully",
            "user_id": req.user_id,
            "genres": result.get("recommended_genres", [])
        }
    except Exception as e:
        # Log the exception for debugging
        print(f"Error in langgraph_flow_endpoint: {e}")
        # Return a generic error response
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.get('/conversation-count/{user_id}')
def get_conversation_count(user_id: str):
    """Get the number of conversations for a user"""
    count = count_conversations(user_id)
    return {"user_id": user_id, "conversation_count": count}

@app.get('/initial-message/{user_id}', response_model=ChatResponse)
def get_initial_message(user_id: str):
    """Get the initial chat history with greeting for a user"""
    from chatbot import get_user_history, generate_initial_response, save_user_history
    
    # Get current history
    history = get_user_history(user_id)
    
    # If this is a new user (only system message), generate initial AI message
    if len(history) == 1 and history[0]['type'] == 'system':
        initial_response = generate_initial_response()
        history.append({'type': 'ai', 'content': initial_response})
        # Save the updated history
        save_user_history(user_id, history)
    
    out_history = [Message(**m) for m in history]
    return ChatResponse(chat_history=out_history)

# ---- MVP (No-LLM, No-Model) deterministic recommendation endpoint ----

# Minimal emotion keyword mapping (rule-based)
EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "joy": ["happy", "joy", "cheerful", "delighted", "good", "great", "excited"],
    "sadness": ["sad", "down", "blue", "depressed", "unhappy", "heartbroken"],
    "anger": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated"],
    "fear": ["afraid", "scared", "anxious", "nervous", "worried", "terrified"],
    "love": ["romantic", "love", "loving", "sweet", "heartwarming"],
    "disgust": ["disgusted", "gross", "nauseated"],
    "surprise": ["surprised", "shocked", "unexpected", "twist"],
    "annoyance": ["annoyed", "bored", "meh"],
    "excitement": ["thrilled", "pumped", "hyped", "excited"],
    "neutral": ["neutral", "fine", "ok", "okay"],
}

EMOTION_TO_GENRES: Dict[str, Dict[str, List[str]]] = {
    "joy": {"resonant": ["Comedy", "Family", "Animation", "Music"], "contrast": ["Horror", "War", "Thriller"]},
    "amusement": {"resonant": ["Comedy", "Animation", "Family"], "contrast": ["Film-Noir", "Crime", "War"]},
    "excitement": {"resonant": ["Action", "Adventure", "Sci-Fi", "Fantasy"], "contrast": ["Documentary", "Drama"]},
    "optimism": {"resonant": ["Adventure", "Romance", "Family"], "contrast": ["Thriller", "War", "Horror"]},
    "pride": {"resonant": ["Biography", "History", "Drama"], "contrast": ["Horror", "Film-Noir"]},
    "gratitude": {"resonant": ["Romance", "Family", "Comedy"], "contrast": ["Horror", "War"]},
    "love": {"resonant": ["Romance", "Drama", "Music"], "contrast": ["Action", "Horror"]},
    "caring": {"resonant": ["Family", "Romance", "Drama"], "contrast": ["Thriller", "War"]},
    "admiration": {"resonant": ["Biography", "History", "Documentary"], "contrast": ["Horror", "Film-Noir"]},
    "relief": {"resonant": ["Comedy", "Romance", "Family"], "contrast": ["Thriller", "Horror"]},
    "satisfaction": {"resonant": ["Drama", "Romance", "Comedy"], "contrast": ["Horror", "Crime"]},
    "approval": {"resonant": ["Comedy", "Family", "Animation"], "contrast": ["War", "Thriller"]},
    "surprise": {"resonant": ["Mystery", "Sci-Fi", "Fantasy"], "contrast": ["Documentary", "Drama"]},
    "curiosity": {"resonant": ["Sci-Fi", "Mystery", "Adventure"], "contrast": ["War", "Action"]},
    "realization": {"resonant": ["Drama", "Documentary", "Mystery"], "contrast": ["Action", "Comedy"]},
    "confusion": {"resonant": ["Mystery", "Drama", "Sci-Fi"], "contrast": ["Comedy", "Family"]},
    "neutral": {"resonant": ["Documentary", "Drama"], "contrast": ["Horror", "Action"]},
    "sadness": {"resonant": ["Drama", "Romance", "Music"], "contrast": ["Comedy", "Action"]},
    "grief": {"resonant": ["History", "Drama", "Biography"], "contrast": ["Comedy", "Animation"]},
    "loneliness": {"resonant": ["Drama", "Romance", "Music"], "contrast": ["Family", "Adventure"]},
    "remorse": {"resonant": ["Drama", "Crime", "History"], "contrast": ["Action", "Comedy"]},
    "anger": {"resonant": ["Action", "Crime", "Thriller"], "contrast": ["Family", "Romance"]},
    "annoyance": {"resonant": ["Comedy", "Crime", "Drama"], "contrast": ["Music", "Animation"]},
    "disapproval": {"resonant": ["Crime", "Drama", "War"], "contrast": ["Romance", "Comedy"]},
    "fear": {"resonant": ["Horror", "Thriller", "Mystery"], "contrast": ["Romance", "Comedy"]},
    "nervousness": {"resonant": ["Thriller", "Drama", "Sci-Fi"], "contrast": ["Comedy", "Family"]},
    "embarrassment": {"resonant": ["Comedy", "Romance", "Drama"], "contrast": ["War", "Horror"]},
    "disgust": {"resonant": ["Film-Noir", "Drama", "Crime"], "contrast": ["Comedy", "Romance"]},
    "disappointment": {"resonant": ["Biography", "Romance", "Drama"], "contrast": ["Animation", "Comedy"]},
}

def _infer_emotions_rule_based(text: str) -> List[str]:
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        scores[emotion] = sum(1 for kw in keywords if kw in text_lower)
    # pick top 1-2 with non-zero scores; fallback to neutral
    non_zero = [(e, s) for e, s in scores.items() if s > 0]
    if not non_zero:
        return ["neutral"]
    non_zero.sort(key=lambda x: x[1], reverse=True)
    top = [e for e, _ in non_zero[:2]]
    return top

def _map_emotions_to_genres(emotions: List[str], preference: str) -> List[str]:
    pref = (preference or "neutral").strip().lower()
    if pref not in {"match", "shift", "neutral"}:
        pref = "neutral"
    resonant_weight = 0.5 if pref == "neutral" else (0.7 if pref == "match" else 0.3)
    contrast_weight = 1.0 - resonant_weight
    genre_scores: Dict[str, float] = {}
    for emotion in emotions:
        mapping = EMOTION_TO_GENRES.get(emotion)
        if not mapping:
            continue
        for g in mapping["resonant"]:
            genre_scores[g] = genre_scores.get(g, 0.0) + resonant_weight
        for g in mapping["contrast"]:
            genre_scores[g] = genre_scores.get(g, 0.0) + contrast_weight
    # top 4 genres
    top = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    return [g for g, _ in top]

@lru_cache(maxsize=1)
def _load_movies_data() -> pd.DataFrame:
    movies_path = os.path.join("database_movielens", "movies.csv")
    ratings_path = os.path.join("database_movielens", "filtered_data.csv")
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    agg = ratings.groupby("movieId").agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count")).reset_index()
    df = movies.merge(agg, on="movieId", how="left")
    df["avg_rating"] = df["avg_rating"].fillna(0.0)
    df["num_ratings"] = df["num_ratings"].fillna(0).astype(int)
    return df

def _recommend_by_genres(genres: List[str], top_n: int) -> List[Dict[str, Any]]:
    df = _load_movies_data()
    # filter by any overlap in genres
    mask = df["genres"].fillna("").apply(lambda s: any(g in s for g in genres))
    candidates = df[mask].copy()
    if candidates.empty:
        # fallback to globally popular
        candidates = df.copy()
    # rank: more ratings first, then higher avg rating
    candidates = candidates.sort_values(["num_ratings", "avg_rating"], ascending=[False, False]).head(top_n)
    out: List[Dict[str, Any]] = []
    for _, row in candidates.iterrows():
        out.append({
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "genres": [g for g in str(row["genres"]).split("|") if g],
            "avg_rating": float(row["avg_rating"]),
            "num_ratings": int(row["num_ratings"]),
        })
    return out

@app.post('/mvp/recommend', response_model=MVPNeedInputResponse | MVPRecommendResponse)
def mvp_recommend(req: MVPRecommendRequest):
    # Step 1: ask for mood_text if missing
    if not req.mood_text or not req.mood_text.strip():
        return MVPNeedInputResponse(
            status="need_input",
            next_question="Tell us about your mood or what you'd like to watch (a sentence is fine).",
            required_field="mood_text",
        )
    # Step 2: ask for preference if missing/invalid
    if not req.preference or req.preference.strip().lower() not in {"match", "shift", "neutral"}:
        return MVPNeedInputResponse(
            status="need_input",
            next_question="Would you like to match your mood, shift it, or keep it neutral? (match/shift/neutral)",
            required_field="preference",
        )
    # Step 3: rule-based emotions -> genres -> movies
    emotions = _infer_emotions_rule_based(req.mood_text)
    top_genres = _map_emotions_to_genres(emotions, req.preference)
    recs = _recommend_by_genres(top_genres, req.top_n)
    return MVPRecommendResponse(
        status="success",
        recommended_genres=top_genres,
        recommendations=[MovieItem(**r) for r in recs],
    )
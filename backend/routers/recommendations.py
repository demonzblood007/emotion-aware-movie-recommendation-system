"""Combined recommendations router - orchestrates all steps for backward compatibility."""

from fastapi import APIRouter, Depends

import torch

from backend.schemas.recommendations import (
    EmotionScore,
    GenreScore,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
)
from backend.services.emotion_service import get_emotion_service
from backend.services.genre_service import get_genre_service
from backend.services.metadata_service import get_metadata_service
from backend.services.ranking_service import get_ranking_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("", response_model=RecommendationResponse)
def create_recommendations(
    payload: RecommendationRequest,
) -> RecommendationResponse:
    """
    Combined endpoint that orchestrates all steps.
    
    This endpoint maintains backward compatibility by calling all granular services
    in the same order as before. The logic is identical, just organized differently.
    """
    # Step 1: Detect emotions
    emotion_service = get_emotion_service()
    emotions = emotion_service.detect_emotions(payload.mood_text)
    
    # Step 2: Map emotions to genres
    genre_service = get_genre_service()
    genre_scores = genre_service.map_emotions_to_genres(emotions, payload.strategy)
    
    # Step 3: Create genre vector
    genre_vector = genre_service.create_genre_vector(genre_scores)
    
    # Step 4: Resolve user
    ranking_service = get_ranking_service()
    user_idx, existing_user = ranking_service.resolve_user_idx(payload.user_id)
    
    # Step 5: Rank movies
    rec_df = ranking_service.rank_movies(
        user_idx=user_idx,
        genre_vector=genre_vector,
        top_n=payload.top_n,
    )
    
    # Step 6: Format and enrich
    results = ranking_service.format_recommendations(rec_df)
    metadata_service = get_metadata_service()
    enriched_results = metadata_service.enrich_batch(results)
    
    return RecommendationResponse(
        user_id=payload.user_id,
        user_idx=user_idx,
        existing_user=existing_user,
        emotions=[
            EmotionScore(emotion=name, score=float(score)) for name, score in emotions
        ],
        genres=[
            GenreScore(genre=name, score=float(score))
            for name, score in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        ],
        results=[RecommendationItem(**item) for item in enriched_results],
    )

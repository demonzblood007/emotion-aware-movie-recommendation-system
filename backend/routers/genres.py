"""Genre mapping router - granular endpoint for emotion-to-genre mapping."""

from fastapi import APIRouter, Depends

from backend.schemas.recommendations import (
    GenreMappingRequest,
    GenreMappingResponse,
    GenreScore,
)
from backend.services.genre_service import GenreService, get_genre_service

router = APIRouter(prefix="/genres", tags=["genres"])


@router.post("/map", response_model=GenreMappingResponse)
def map_emotions_to_genres(
    payload: GenreMappingRequest,
    service: GenreService = Depends(get_genre_service),
) -> GenreMappingResponse:
    """
    Map detected emotions to genre scores.
    
    This is a granular endpoint that maps emotions to genres based on strategy.
    Use this for agentic workflows or when you need genre mapping separately.
    """
    # Convert EmotionScore objects to tuples
    emotions = [(e.emotion, e.score) for e in payload.emotions]
    
    genre_scores = service.map_emotions_to_genres(emotions, payload.strategy)
    
    return GenreMappingResponse(
        genres=[
            GenreScore(genre=name, score=float(score))
            for name, score in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        ],
        genre_vector=None,  # Genre vector is tensor, not JSON-serializable
    )


@router.post("/vector", response_model=dict)
def create_genre_vector(
    payload: GenreMappingRequest,
    service: GenreService = Depends(get_genre_service),
) -> dict:
    """
    Create a genre vector from emotions.
    
    This creates the actual genre vector used for ranking.
    """
    # Convert EmotionScore objects to tuples
    emotions = [(e.emotion, e.score) for e in payload.emotions]
    
    genre_scores = service.map_emotions_to_genres(emotions, payload.strategy)
    genre_vector = service.create_genre_vector(genre_scores)
    
    # Convert tensor to list for JSON serialization
    genre_vector_list = genre_vector.cpu().tolist()
    
    return {"genre_vector": genre_vector_list}


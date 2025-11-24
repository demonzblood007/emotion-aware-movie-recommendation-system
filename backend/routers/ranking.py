"""Movie ranking router - granular endpoint for movie ranking."""

from fastapi import APIRouter, Depends

import torch

from backend.schemas.recommendations import (
    RankingRequest,
    RankingResponse,
    RecommendationItem,
)
from backend.services.metadata_service import get_metadata_service
from backend.services.ranking_service import RankingService, get_ranking_service

router = APIRouter(prefix="/ranking", tags=["ranking"])


@router.post("/rank", response_model=RankingResponse)
def rank_movies(
    payload: RankingRequest,
    service: RankingService = Depends(get_ranking_service),
) -> RankingResponse:
    """
    Rank movies based on user and genre vector.
    
    This is a granular endpoint that performs movie ranking.
    Use this for agentic workflows or when you need ranking separately.
    """
    # Convert list to tensor
    genre_vector = torch.tensor(payload.genre_vector, dtype=torch.float32)
    
    # Rank movies
    rec_df = service.rank_movies(
        user_idx=payload.user_idx,
        genre_vector=genre_vector,
        top_n=payload.top_n,
    )
    
    # Format results
    results = service.format_recommendations(rec_df)
    
    # Enrich with metadata
    metadata_service = get_metadata_service()
    enriched_results = metadata_service.enrich_batch(results)
    
    return RankingResponse(
        results=[
            RecommendationItem(**item) for item in enriched_results
        ]
    )


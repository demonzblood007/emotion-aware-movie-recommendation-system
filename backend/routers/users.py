"""User resolution router - granular endpoint for user ID resolution."""

from fastapi import APIRouter, Depends

from backend.schemas.recommendations import (
    UserResolutionRequest,
    UserResolutionResponse,
)
from backend.services.ranking_service import RankingService, get_ranking_service

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/resolve", response_model=UserResolutionResponse)
def resolve_user(
    payload: UserResolutionRequest,
    service: RankingService = Depends(get_ranking_service),
) -> UserResolutionResponse:
    """
    Resolve external user ID to internal index.
    
    This is a granular endpoint for user resolution.
    Use this for agentic workflows or when you need user resolution separately.
    """
    user_idx, existing_user = service.resolve_user_idx(payload.user_id)
    
    return UserResolutionResponse(
        user_idx=user_idx,
        existing_user=existing_user,
        user_id=payload.user_id,
    )


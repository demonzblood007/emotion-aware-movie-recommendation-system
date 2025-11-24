"""Emotion detection router - granular endpoint for emotion detection."""

from fastapi import APIRouter, Depends

from backend.schemas.recommendations import (
    EmotionDetectionRequest,
    EmotionDetectionResponse,
    EmotionScore,
)
from backend.services.emotion_service import EmotionService, get_emotion_service

router = APIRouter(prefix="/emotions", tags=["emotions"])


@router.post("/detect", response_model=EmotionDetectionResponse)
def detect_emotions(
    payload: EmotionDetectionRequest,
    service: EmotionService = Depends(get_emotion_service),
) -> EmotionDetectionResponse:
    """
    Detect emotions from text using BERT model.
    
    This is a granular endpoint that only performs emotion detection.
    Use this for agentic workflows or when you only need emotion analysis.
    """
    emotions = service.detect_emotions(payload.text, top_k=payload.top_k)
    
    return EmotionDetectionResponse(
        emotions=[
            EmotionScore(emotion=name, score=float(score)) for name, score in emotions
        ]
    )


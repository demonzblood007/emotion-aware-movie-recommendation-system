"""Emotion detection service - granular service for emotion detection step."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import final_recommender as fr
from backend.config import get_settings


class EmotionService:
    """Service for detecting emotions from text using BERT model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._bootstrap_model()

    def _bootstrap_model(self) -> None:
        """Load emotion detection model."""
        fr.EMOTION_MODEL_PATH = self.settings.emotion_model_path
        (
            self.emotion_model,
            self.tokenizer,
            self.emotion_labels,
            self.device,
        ) = fr.load_emotion_model(self.settings.emotion_model_path)

    def detect_emotions(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Detect emotions from text using BERT model.
        
        Args:
            text: Input text to analyze
            top_k: Number of top emotions to return
            
        Returns:
            List of (emotion, score) tuples
        """
        return fr.predict_emotions(
            text,
            self.emotion_model,
            self.tokenizer,
            self.emotion_labels,
            self.device,
            top_k=top_k,
        )


@lru_cache(maxsize=1)
def get_emotion_service() -> EmotionService:
    """Cached singleton instance."""
    return EmotionService()


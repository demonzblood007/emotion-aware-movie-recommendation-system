"""Genre mapping service - granular service for emotion-to-genre mapping step."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import final_recommender as fr
from backend.config import get_settings


class GenreService:
    """Service for mapping emotions to genres and creating genre vectors."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._bootstrap_data()

    def _bootstrap_data(self) -> None:
        """Load genre mapping data."""
        (
            self.data,
            self._genre_map,
            self.user_map,
            self.movie_map,
            self.reverse_movie_map,
        ) = fr.load_movies_data()

    def map_emotions_to_genres(
        self, emotions: List[Tuple[str, float]], strategy: str = "neutral"
    ) -> Dict[str, float]:
        """
        Map detected emotions to genre scores.
        
        Args:
            emotions: List of (emotion, score) tuples
            strategy: "match", "shift", or "neutral"
            
        Returns:
            Dictionary mapping genre names to scores
        """
        genre_scores = fr.map_emotions_to_genres(emotions, fr.emotion_to_genres, strategy)
        if not genre_scores:
            # Fallback to default genres
            genre_scores = {
                genre: 1.0
                for genre in ["Drama", "Comedy", "Documentary", "Mystery", "Action"]
            }
        return genre_scores

    def create_genre_vector(
        self, genre_scores: Dict[str, float], top_n: int = 4
    ) -> fr.torch.Tensor:
        """
        Create a genre vector from genre scores.
        
        Args:
            genre_scores: Dictionary mapping genre names to scores
            top_n: Number of top genres to include in vector
            
        Returns:
            PyTorch tensor representing genre vector
        """
        return fr.create_genre_vector(genre_scores, self._genre_map, top_n=top_n)

    @property
    def genre_map(self) -> Dict[str, int]:
        """Get the genre mapping dictionary."""
        return self._genre_map
    
    def get_genre_map(self) -> Dict[str, int]:
        """Get the genre mapping dictionary."""
        return self._genre_map


@lru_cache(maxsize=1)
def get_genre_service() -> GenreService:
    """Cached singleton instance."""
    return GenreService()


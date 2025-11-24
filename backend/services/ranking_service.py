"""Movie ranking service - granular service for NCF-based movie ranking."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import pandas as pd
import final_recommender as fr
from backend.config import get_settings


class RankingService:
    """Service for ranking movies using NCF model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._bootstrap_models()

    def _bootstrap_models(self) -> None:
        """Load ranking models and data."""
        fr.MOVIE_MODEL_PATH = self.settings.movie_model_path
        fr.MOVIE_DATA_PATH = self.settings.movielens_filtered_csv

        movies_df = pd.read_csv(self.settings.movies_csv)
        fr.movies_df = movies_df
        fr.movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
        self.movie_id_to_title = fr.movie_id_to_title
        self.movie_id_to_genres = dict(
            zip(movies_df["movieId"], movies_df["genres"].fillna(""))
        )

        (
            self.data,
            self.genre_map,
            self.user_map,
            self.movie_map,
            self.reverse_movie_map,
        ) = fr.load_movies_data()

        (
            self.movie_model,
            self.trained_num_users,
            self.trained_num_movies,
        ) = fr.load_movie_model(
            self.settings.movie_model_path,
            num_users=len(self.user_map),
            num_movies=len(self.movie_map),
            num_genres=len(self.genre_map),
        )

        self.data = self.data[(self.data["movie_idx"] < self.trained_num_movies)]

        # Get device from emotion service (they share the same device)
        from backend.services.emotion_service import get_emotion_service

        emotion_service = get_emotion_service()
        self.device = emotion_service.device

    def resolve_user_idx(self, user_id: int | None) -> Tuple[int, bool]:
        """
        Resolve external user ID to internal index.
        
        Args:
            user_id: External user ID (MovieLens ID) or None
            
        Returns:
            Tuple of (internal_user_idx, is_existing_user)
        """
        if user_id is None:
            return 0, False
        if user_id in self.user_map:
            idx = self.user_map[user_id]
            if idx < self.trained_num_users:
                return idx, True
        return 0, False

    def rank_movies(
        self,
        user_idx: int,
        genre_vector: fr.torch.Tensor,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Rank movies for a user based on genre vector.
        
        Args:
            user_idx: Internal user index
            genre_vector: Genre vector tensor
            top_n: Number of top recommendations to return
            
        Returns:
            DataFrame with ranked movies (movieId, genres, predicted_rating)
        """
        return fr.recommend_movies(
            user_idx,
            genre_vector,
            self.movie_model,
            self.data,
            self.genre_map,
            top_n,
            device=self.device,
        )

    def format_recommendations(self, rec_df: pd.DataFrame) -> List[dict]:
        """
        Format recommendation DataFrame into list of dictionaries.
        
        Args:
            rec_df: DataFrame with recommendations
            
        Returns:
            List of recommendation dictionaries (without metadata)
        """
        results: List[dict] = []
        for rank, (_, row) in enumerate(rec_df.iterrows(), start=1):
            movie_id = int(row["movieId"])
            title = self.movie_id_to_title.get(movie_id, "Unknown Title")
            genres = (
                row["genres"]
                if isinstance(row["genres"], str)
                else self.movie_id_to_genres.get(movie_id, "")
            )
            genre_list = [g for g in genres.split("|") if g]
            results.append(
                {
                    "rank": rank,
                    "movie_id": movie_id,
                    "title": title,
                    "genres": genre_list,
                    "predicted_rating": round(float(row["predicted_rating"]), 2),
                }
            )
        return results


@lru_cache(maxsize=1)
def get_ranking_service() -> RankingService:
    """Cached singleton instance."""
    return RankingService()


from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import pandas as pd

from backend.config import get_settings
from backend.services.metadata_service import get_metadata_service

import final_recommender as fr


class NCFService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._bootstrap_models()

    def _bootstrap_models(self) -> None:
        fr.MOVIE_MODEL_PATH = self.settings.movie_model_path
        fr.EMOTION_MODEL_PATH = self.settings.emotion_model_path
        fr.MOVIE_DATA_PATH = self.settings.movielens_filtered_csv

        movies_df = pd.read_csv(self.settings.movies_csv)
        fr.movies_df = movies_df
        fr.movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
        self.movie_id_to_title = fr.movie_id_to_title
        self.movie_id_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"].fillna("")))

        (
            self.data,
            self.genre_map,
            self.user_map,
            self.movie_map,
            self.reverse_movie_map,
        ) = fr.load_movies_data()

        self.emotion_model, self.tokenizer, self.emotion_labels, self.device = fr.load_emotion_model(
            self.settings.emotion_model_path
        )
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
        self.data = self.data[
            (self.data["movie_idx"] < self.trained_num_movies)
        ]

    def _resolve_user_idx(self, user_id: int | None) -> tuple[int, bool]:
        if user_id is None:
            return 0, False
        if user_id in self.user_map:
            idx = self.user_map[user_id]
            if idx < self.trained_num_users:
                return idx, True
        return 0, False

    def generate_recommendations(
        self,
        *,
        user_id: int | None,
        mood_text: str,
        strategy: str,
        top_n: int,
    ) -> Dict:
        emotions = fr.predict_emotions(
            mood_text,
            self.emotion_model,
            self.tokenizer,
            self.emotion_labels,
            self.device,
        )
        genre_scores = fr.map_emotions_to_genres(emotions, fr.emotion_to_genres, strategy)
        if not genre_scores:
            genre_scores = {genre: 1.0 for genre in ["Drama", "Comedy", "Documentary", "Mystery", "Action"]}

        genre_vector = fr.create_genre_vector(genre_scores, self.genre_map)
        user_idx, resolved = self._resolve_user_idx(user_id)

        rec_df = fr.recommend_movies(
            user_idx,
            genre_vector,
            self.movie_model,
            self.data,
            self.genre_map,
            top_n,
            device=self.device,
        )
        recs = self._format_recommendations(rec_df)

        return {
            "user_id": user_id,
            "user_idx": user_idx,
            "existing_user": resolved,
            "emotions": [
                {"emotion": name, "score": float(score)} for name, score in emotions
            ],
            "genres": [
                {"genre": name, "score": float(score)} for name, score in sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
            ],
            "results": recs,
        }

    def _format_recommendations(self, rec_df) -> List[Dict]:
        results: List[Dict] = []
        for rank, (_, row) in enumerate(rec_df.iterrows(), start=1):
            movie_id = int(row["movieId"])
            title = self.movie_id_to_title.get(movie_id, "Unknown Title")
            genres = row["genres"] if isinstance(row["genres"], str) else self.movie_id_to_genres.get(movie_id, "")
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
        
        # Enrich with metadata (posters, ratings, watch links)
        metadata_service = get_metadata_service()
        enriched_results = metadata_service.enrich_batch(results)
        return enriched_results


@lru_cache(maxsize=1)
def get_ncf_service() -> NCFService:
    return NCFService()

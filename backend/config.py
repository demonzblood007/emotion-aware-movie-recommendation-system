from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = Field(default="development", alias="ENV")
    movie_model_path: str = Field(default="./best_model.pt", alias="MOVIE_MODEL_PATH")
    emotion_model_path: str = Field(
        default="SamLowe/roberta-base-go_emotions",
        alias="EMOTION_MODEL_PATH",
    )
    movielens_filtered_csv: str = Field(
        default="./database_movielens/filtered_data.csv", alias="MOVIELENS_FILTERED_CSV"
    )
    movies_csv: str = Field(default="./database_movielens/movies.csv", alias="MOVIES_CSV")
    tmdb_api_key: str | None = Field(default=None, alias="TMDB_API_KEY")
    omdb_api_key: str | None = Field(default=None, alias="OMDB_API_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor so every import shares one Settings instance."""
    return Settings()

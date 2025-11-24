from typing import List, Literal, Optional

from pydantic import BaseModel, Field


StrategyLiteral = Literal["match", "shift", "neutral"]


class RecommendationRequest(BaseModel):
    mood_text: str = Field(..., min_length=3, description="Free-form text describing the user's mood")
    user_id: Optional[int] = Field(None, description="Existing MovieLens user id if available")
    strategy: StrategyLiteral = Field("neutral", description="Whether to match, shift, or keep the mood neutral")
    top_n: int = Field(10, ge=1, le=20)


class EmotionScore(BaseModel):
    emotion: str
    score: float


class GenreScore(BaseModel):
    genre: str
    score: float


class WatchProvider(BaseModel):
    name: str
    type: str  # "flatrate", "rent", "buy"
    logo: Optional[str] = None
    url: Optional[str] = None  # Link to where to watch (JustWatch or provider direct)


class RecommendationItem(BaseModel):
    rank: int
    movie_id: int
    title: str
    genres: List[str]
    predicted_rating: float
    # Metadata fields (optional, may be None if enrichment fails)
    poster_url: Optional[str] = None
    imdb_rating: Optional[float] = None
    tmdb_rating: Optional[float] = None
    imdb_url: Optional[str] = None
    tmdb_url: Optional[str] = None
    watch_providers: Optional[List[WatchProvider]] = None


class RecommendationResponse(BaseModel):
    user_id: Optional[int]
    user_idx: int
    existing_user: bool
    emotions: List[EmotionScore]
    genres: List[GenreScore]
    results: List[RecommendationItem]


# Granular endpoint schemas
class EmotionDetectionRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Text to analyze for emotions")
    top_k: int = Field(3, ge=1, le=10, description="Number of top emotions to return")


class EmotionDetectionResponse(BaseModel):
    emotions: List[EmotionScore]


class GenreMappingRequest(BaseModel):
    emotions: List[EmotionScore] = Field(..., description="List of detected emotions")
    strategy: StrategyLiteral = Field("neutral", description="Mapping strategy")


class GenreMappingResponse(BaseModel):
    genres: List[GenreScore]
    genre_vector: Optional[List[float]] = Field(None, description="Genre vector representation")


class UserResolutionRequest(BaseModel):
    user_id: Optional[int] = Field(None, description="External user ID to resolve")


class UserResolutionResponse(BaseModel):
    user_idx: int
    existing_user: bool
    user_id: Optional[int]


class RankingRequest(BaseModel):
    user_idx: int = Field(..., description="Internal user index")
    genre_vector: List[float] = Field(..., description="Genre vector")
    top_n: int = Field(10, ge=1, le=20, description="Number of recommendations")


class RankingResponse(BaseModel):
    results: List[RecommendationItem]

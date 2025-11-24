export type Strategy = "match" | "neutral" | "shift";

export interface HealthResponse {
  status: string;
  env?: string;
}

export interface RecommendationPayload {
  mood_text: string;
  strategy: Strategy;
  top_n: number;
  user_id?: number;
}

export interface EmotionScore {
  emotion: string;
  score: number;
}

export interface GenreScore {
  genre: string;
  score: number;
}

export interface WatchProvider {
  name: string;
  type: string; // "flatrate" | "rent" | "buy"
  logo?: string | null;
  url?: string | null; // Link to where to watch
}

export interface RecommendationItem {
  rank: number;
  movie_id: number;
  title: string;
  genres: string[];
  predicted_rating: number;
  // Metadata fields (optional)
  poster_url?: string | null;
  imdb_rating?: number | null;
  tmdb_rating?: number | null;
  imdb_url?: string | null;
  tmdb_url?: string | null;
  watch_providers?: WatchProvider[] | null;
}

export interface RecommendationResponse {
  user_id?: number | null;
  user_idx: number;
  existing_user: boolean;
  emotions: EmotionScore[];
  genres: GenreScore[];
  results: RecommendationItem[];
}

// Granular endpoint types
export interface EmotionDetectionRequest {
  text: string;
  top_k?: number;
}

export interface EmotionDetectionResponse {
  emotions: EmotionScore[];
}

export interface GenreMappingRequest {
  emotions: EmotionScore[];
  strategy: Strategy;
}

export interface GenreMappingResponse {
  genres: GenreScore[];
  genre_vector?: number[] | null;
}

export interface UserResolutionRequest {
  user_id?: number | null;
}

export interface UserResolutionResponse {
  user_idx: number;
  existing_user: boolean;
  user_id?: number | null;
}

export interface RankingRequest {
  user_idx: number;
  genre_vector: number[];
  top_n: number;
}

export interface RankingResponse {
  results: RecommendationItem[];
}




import { API_ROUTES } from "./config";
import {
  EmotionDetectionRequest,
  EmotionDetectionResponse,
  GenreMappingRequest,
  GenreMappingResponse,
  HealthResponse,
  RankingRequest,
  RankingResponse,
  RecommendationPayload,
  RecommendationResponse,
  UserResolutionRequest,
  UserResolutionResponse,
} from "./types";

const withErrorHandling = async <T>(res: Response, url: string): Promise<T> => {
  const contentType = res.headers.get("content-type") || "";
  
  // Check if response is HTML (error page) instead of JSON
  if (contentType.includes("text/html")) {
    throw new Error(`API returned HTML instead of JSON. URL: ${url}. Check if backend is running and URL is correct.`);
  }
  
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail =
        typeof body?.detail === "string"
          ? body.detail
          : JSON.stringify(body ?? {});
    } catch {
      detail = res.statusText;
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  
  // Check content type before parsing JSON
  if (contentType && !contentType.includes("application/json")) {
    throw new Error(`API returned ${contentType} instead of JSON. URL: ${url}`);
  }
  
  try {
    return await res.json() as Promise<T>;
  } catch (e) {
    throw new Error(`Invalid JSON response from API: ${url}`);
  }
};

export const fetchHealth = () => {
  const url = API_ROUTES.health;
  return fetch(url, {
    cache: "no-store",
    mode: "cors",
    credentials: "omit",
    headers: {
      "Accept": "application/json",
      "ngrok-skip-browser-warning": "true", // Bypass ngrok interstitial page
    },
  }).then((res) => withErrorHandling<HealthResponse>(res, url)).catch((error) => {
    if (error instanceof Error) {
      console.error("[API] Health check failed:", error.message);
      console.error("[API] Attempted URL:", url);
      console.error("[API] Tip: If you see HTML, ngrok might be showing an interstitial page.");
      console.error("[API] Try opening the URL directly in browser first to bypass it.");
    }
    throw error;
  });
};

// Combined endpoint (backward compatibility)
export const requestRecommendations = (payload: RecommendationPayload) => {
  const url = API_ROUTES.recommendations;
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true", // Bypass ngrok interstitial page
    },
    body: JSON.stringify(payload),
  }).then((res) => withErrorHandling<RecommendationResponse>(res, url));
};

// Granular endpoints
export const detectEmotions = (payload: EmotionDetectionRequest) => {
  const url = API_ROUTES.emotions.detect;
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify(payload),
  }).then((res) => withErrorHandling<EmotionDetectionResponse>(res, url));
};

export const mapEmotionsToGenres = (payload: GenreMappingRequest) => {
  const url = API_ROUTES.genres.map;
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify(payload),
  }).then((res) => withErrorHandling<GenreMappingResponse>(res, url));
};

export const createGenreVector = (payload: GenreMappingRequest) => {
  const url = API_ROUTES.genres.vector;
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify(payload),
  }).then((res) => withErrorHandling<{ genre_vector: number[] }>(res, url));
};

export const resolveUser = (payload: UserResolutionRequest) => {
  const url = API_ROUTES.users.resolve;
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify(payload),
  }).then((res) => withErrorHandling<UserResolutionResponse>(res, url));
};

export const rankMovies = (payload: RankingRequest) => {
  const url = API_ROUTES.ranking.rank;
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify(payload),
  }).then((res) => withErrorHandling<RankingResponse>(res, url));
};

// Orchestrated recommendation function using granular endpoints
// This function can be called with an optional progress callback
export const requestRecommendationsGranular = async (
  payload: RecommendationPayload,
  onProgress?: (step: string) => void
): Promise<RecommendationResponse> => {
  // Step 1: Detect emotions
  onProgress?.("Analyzing your mood...");
  const emotionResponse = await detectEmotions({
    text: payload.mood_text,
    top_k: 3,
  });

  // Step 2: Map emotions to genres and create vector
  onProgress?.("Finding perfect matches...");
  const genreVectorResponse = await createGenreVector({
    emotions: emotionResponse.emotions,
    strategy: payload.strategy,
  });

  if (!genreVectorResponse.genre_vector) {
    throw new Error("Failed to create genre vector");
  }

  // Step 3: Resolve user
  onProgress?.("Loading your preferences...");
  const userResponse = await resolveUser({
    user_id: payload.user_id,
  });

  // Step 4: Rank movies
  onProgress?.("Ranking recommendations...");
  const rankingResponse = await rankMovies({
    user_idx: userResponse.user_idx,
    genre_vector: genreVectorResponse.genre_vector,
    top_n: payload.top_n,
  });

  // Step 5: Get genre scores for response
  onProgress?.("Enriching with details...");
  const genreMappingResponse = await mapEmotionsToGenres({
    emotions: emotionResponse.emotions,
    strategy: payload.strategy,
  });

  // Combine into full response
  return {
    user_id: payload.user_id,
    user_idx: userResponse.user_idx,
    existing_user: userResponse.existing_user,
    emotions: emotionResponse.emotions,
    genres: genreMappingResponse.genres,
    results: rankingResponse.results,
  };
};




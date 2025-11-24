const FALLBACK_API_URL = "http://localhost:8000";

// Get API URL and remove trailing slash if present
let rawApiUrl =
  process.env.NEXT_PUBLIC_RECOMMENDER_API_URL ??
  process.env.RECOMMENDER_API_URL ??
  FALLBACK_API_URL;

// Remove trailing slash
export const API_URL = rawApiUrl.replace(/\/$/, "");

// Debug logging (only in browser, not during SSR, and only once)
if (typeof window !== "undefined" && !(window as any).__API_CONFIG_LOGGED) {
  console.log("[API Config] Using API URL:", API_URL);
  (window as any).__API_CONFIG_LOGGED = true;
}

export const API_ROUTES = {
  health: `${API_URL}/health`,
  recommendations: `${API_URL}/recommendations`,
  // Granular endpoints
  emotions: {
    detect: `${API_URL}/emotions/detect`,
  },
  genres: {
    map: `${API_URL}/genres/map`,
    vector: `${API_URL}/genres/vector`,
  },
  users: {
    resolve: `${API_URL}/users/resolve`,
  },
  ranking: {
    rank: `${API_URL}/ranking/rank`,
  },
};




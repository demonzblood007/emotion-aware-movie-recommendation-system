"""Service for enriching movie recommendations with posters, ratings, and watch links."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Dict, Optional

import httpx
import pandas as pd

from backend.config import get_settings


class MetadataService:
    """Fetches movie metadata from TMDB API and constructs IMDB/watch links."""

    def __init__(self) -> None:
        self.settings = get_settings()
        # Try both environment variable and settings
        self.tmdb_api_key = os.getenv("TMDB_API_KEY") or self.settings.tmdb_api_key
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.tmdb_image_base = "https://image.tmdb.org/t/p/w500"
        self._load_links_data()
        self._init_provider_url_map()
        
        # Rate limiting: TMDB allows 40 requests per 10 seconds
        # Add small delay between requests to stay under limit
        self.tmdb_delay = 0.25  # 250ms delay = ~4 requests/second = safe margin
        self.last_tmdb_request = 0.0
        
        # OMDB rate limiting: 1000 requests/day (free tier)
        # Add delay to be safe
        self.omdb_delay = 0.1  # 100ms delay
        self.last_omdb_request = 0.0
        
        # Debug: Log if API key is missing
        if not self.tmdb_api_key:
            print("Warning: TMDB_API_KEY not found. Posters and watch providers will not be available.")
    
    def _init_provider_url_map(self) -> None:
        """Initialize provider filtering and direct URL construction."""
        import urllib.parse
        
        # Map TMDB provider names to our display names and URLs
        # This dictionary maps various TMDB provider name variations to our standardized names
        self.provider_mapping = {
            # Netflix variations
            "netflix": "Netflix",
            "Netflix": "Netflix",
            
            # Amazon Prime variations
            "amazon prime video": "Amazon Prime",
            "Amazon Prime Video": "Amazon Prime",
            "amazon prime": "Amazon Prime",
            "Amazon Prime": "Amazon Prime",
            
            # Disney+ Hotstar variations
            "disney plus": "Disney+ Hotstar",
            "Disney Plus": "Disney+ Hotstar",
            "disney+": "Disney+ Hotstar",
            "Disney+": "Disney+ Hotstar",
            "disney+ hotstar": "Disney+ Hotstar",
            "Disney+ Hotstar": "Disney+ Hotstar",
            "hotstar": "Disney+ Hotstar",
            "Hotstar": "Disney+ Hotstar",
            
            # YouTube variations
            "youtube": "YouTube",
            "YouTube": "YouTube",
            "youtube premium": "YouTube",
            "YouTube Premium": "YouTube",
            
            # Apple TV variations
            "apple tv plus": "Apple TV",
            "Apple TV Plus": "Apple TV",
            "apple tv+": "Apple TV",
            "Apple TV+": "Apple TV",
            "apple tv": "Apple TV",
            "Apple TV": "Apple TV",
        }
    
    def _construct_provider_url(self, provider_name: str, movie_title: str) -> str:
        """
        Construct direct URL to watch the movie on the specific provider.
        Uses provider-specific search pages for direct access.
        """
        import urllib.parse
        
        # Encode movie title for URL
        encoded_title = urllib.parse.quote_plus(movie_title)
        
        # Direct links for each provider
        provider_urls = {
            "Netflix": f"https://www.netflix.com/search?q={encoded_title}",
            "Amazon Prime Video": f"https://www.amazon.com/s?k={encoded_title}&i=prime-instant-video",
            "Amazon Prime": f"https://www.amazon.com/s?k={encoded_title}&i=prime-instant-video",
            "Disney Plus": f"https://www.hotstar.com/in/search?q={encoded_title}",
            "Disney+ Hotstar": f"https://www.hotstar.com/in/search?q={encoded_title}",
            "Hotstar": f"https://www.hotstar.com/in/search?q={encoded_title}",
            "YouTube": f"https://www.youtube.com/results?search_query={encoded_title}+movie",
            "Apple TV Plus": f"https://tv.apple.com/us/search?term={encoded_title}",
            "Apple TV": f"https://tv.apple.com/us/search?term={encoded_title}",
            "Apple TV+": f"https://tv.apple.com/us/search?term={encoded_title}",
        }
        
        # Return direct link or fallback to Netflix search
        return provider_urls.get(provider_name, f"https://www.netflix.com/search?q={encoded_title}")

    def _load_links_data(self) -> None:
        """Load links.csv to map movieId -> imdbId, tmdbId."""
        links_path = self.settings.movies_csv.replace("movies.csv", "links.csv")
        try:
            links_df = pd.read_csv(links_path)
            # Handle missing values - some movies may not have TMDB/IMDB IDs
            links_df = links_df.fillna({"imdbId": "", "tmdbId": ""})
            # Convert to string for IMDB ID (some may have leading zeros)
            links_df["imdbId"] = links_df["imdbId"].astype(str).str.replace(".0", "", regex=False)
            links_df["tmdbId"] = links_df["tmdbId"].astype(str).str.replace(".0", "", regex=False)
            # Create lookup dicts
            self.movie_id_to_imdb: Dict[int, str] = dict(
                zip(links_df["movieId"], links_df["imdbId"])
            )
            self.movie_id_to_tmdb: Dict[int, str] = dict(
                zip(links_df["movieId"], links_df["tmdbId"])
            )
        except FileNotFoundError:
            print(f"Warning: {links_path} not found. Metadata enrichment will be limited.")
            self.movie_id_to_imdb = {}
            self.movie_id_to_tmdb = {}

    def _fetch_tmdb_details(self, tmdb_id: str, retries: int = 3) -> Optional[Dict]:
        """Fetch movie details from TMDB API with rate limiting and retry logic."""
        if not self.tmdb_api_key or not tmdb_id or tmdb_id == "" or tmdb_id == "nan":
            return None

        # Rate limiting: ensure we don't exceed TMDB's limits
        current_time = time.time()
        time_since_last = current_time - self.last_tmdb_request
        if time_since_last < self.tmdb_delay:
            time.sleep(self.tmdb_delay - time_since_last)
        self.last_tmdb_request = time.time()

        for attempt in range(retries):
            try:
                # Fetch movie details with watch providers
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(
                        f"{self.tmdb_base_url}/movie/{tmdb_id}",
                        params={"api_key": self.tmdb_api_key, "append_to_response": "watch/providers"},
                    )
                
                # Handle rate limiting (429 Too Many Requests)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    if attempt < retries - 1:
                        print(f"TMDB rate limit hit for {tmdb_id}, waiting {retry_after}s before retry {attempt + 1}/{retries}")
                        time.sleep(retry_after)
                        continue
                    else:
                        print(f"TMDB rate limit exceeded for {tmdb_id} after {retries} attempts")
                        return None
                
                # Handle 404 (movie not found) - don't retry
                if response.status_code == 404:
                    # Movie doesn't exist in TMDB, not an error we can fix
                    return None
                
                response.raise_for_status()
                data = response.json()
                return data
                    
            except httpx.HTTPStatusError as e:
                # Don't retry on 404 (movie not found)
                if e.response.status_code == 404:
                    return None
                # Retry on other HTTP errors
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s
                    print(f"TMDB API HTTP error for movie {tmdb_id} (attempt {attempt + 1}/{retries}): {e.response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"TMDB API HTTP error for movie {tmdb_id}: {e.response.status_code} - {e.response.text[:100]}")
                    return None
            except (httpx.HTTPError, KeyError, ValueError) as e:
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    print(f"TMDB API error for {tmdb_id} (attempt {attempt + 1}/{retries}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"TMDB API error for {tmdb_id}: {e}")
                    return None

        return None

    def _get_imdb_rating(self, imdb_id: str, retries: int = 2) -> Optional[float]:
        """Fetch IMDB rating using OMDB API (free tier, 1000 requests/day) with rate limiting."""
        if not imdb_id or not imdb_id.isdigit():
            print(f"[Metadata] Invalid IMDB ID format: {imdb_id}")
            return None

        # Try both environment variable and settings (settings loads from .env file)
        omdb_api_key = os.getenv("OMDB_API_KEY") or self.settings.omdb_api_key
        if not omdb_api_key:
            print(f"[Metadata] OMDB_API_KEY not found in environment or .env file. IMDB ratings will not be available.")
            return None

        # Rate limiting: add small delay between OMDB requests
        current_time = time.time()
        time_since_last = current_time - self.last_omdb_request
        if time_since_last < self.omdb_delay:
            time.sleep(self.omdb_delay - time_since_last)
        self.last_omdb_request = time.time()

        for attempt in range(retries):
            try:
                # Pad IMDB ID to 7 digits (e.g., "123" -> "0000123")
                padded_id = imdb_id.zfill(7)
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(
                        "http://www.omdbapi.com/",
                        params={"i": f"tt{padded_id}", "apikey": omdb_api_key},
                    )
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < retries - 1:
                        wait_time = 2.0  # OMDB rate limit, wait 2 seconds
                        print(f"[Metadata] OMDB rate limit hit for {imdb_id}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[Metadata] OMDB rate limit hit for {imdb_id} after {retries} attempts. Skipping.")
                        return None
                
                response.raise_for_status()
                data = response.json()
                
                # Check for API errors in response
                if data.get("Response") == "False":
                    error_msg = data.get("Error", "Unknown error")
                    print(f"[Metadata] OMDB API error for IMDB ID {imdb_id}: {error_msg}")
                    return None
                
                if data.get("Response") == "True" and "imdbRating" in data:
                    rating_str = data["imdbRating"]
                    if rating_str and rating_str != "N/A":
                        rating = float(rating_str)
                        print(f"[Metadata] IMDB rating fetched successfully for {imdb_id}: {rating}")
                        return rating
                    else:
                        print(f"[Metadata] IMDB rating not available (N/A) for IMDB ID {imdb_id}")
                        return None
                else:
                    print(f"[Metadata] IMDB rating field missing in OMDB response for IMDB ID {imdb_id}")
                    return None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    print(f"[Metadata] OMDB: Movie {imdb_id} not found (404). Skipping.")
                    return None  # Movie not found, don't retry
                if e.response.status_code == 429 and attempt < retries - 1:
                    wait_time = 2.0
                    print(f"[Metadata] OMDB rate limit for {imdb_id}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                # Log HTTP errors
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    print(f"[Metadata] OMDB API HTTP error for {imdb_id} (attempt {attempt + 1}/{retries}): {e.response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_text = e.response.text[:200] if hasattr(e.response, 'text') else str(e)
                    print(f"[Metadata] OMDB API HTTP error for {imdb_id} after {retries} attempts: {e.response.status_code} - {error_text}")
                    return None
            except (httpx.HTTPError, ValueError, KeyError) as e:
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    print(f"[Metadata] OMDB API error for {imdb_id} (attempt {attempt + 1}/{retries}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[Metadata] OMDB API error for {imdb_id} after {retries} attempts: {e}")
                    return None
        return None

    def enrich_movie(
        self, movie_id: int, title: str
    ) -> Dict[str, Optional[str | float]]:
        """
        Enrich a single movie with metadata.

        Returns:
            Dict with: poster_url, imdb_rating, tmdb_rating, imdb_url, tmdb_url, watch_providers
        """
        result: Dict[str, Optional[str | float]] = {
            "poster_url": None,
            "imdb_rating": None,
            "tmdb_rating": None,
            "imdb_url": None,
            "tmdb_url": None,
            "watch_providers": None,
        }

        imdb_id = self.movie_id_to_imdb.get(movie_id, "")
        tmdb_id = self.movie_id_to_tmdb.get(movie_id, "")

        # Construct IMDB URL
        if imdb_id and imdb_id.isdigit():
            padded_id = imdb_id.zfill(7)
            result["imdb_url"] = f"https://www.imdb.com/title/tt{padded_id}/"

        # Construct TMDB URL
        if tmdb_id:
            result["tmdb_url"] = f"https://www.themoviedb.org/movie/{tmdb_id}"

        # Fetch TMDB details (poster, watch providers)
        if tmdb_id and tmdb_id != "" and tmdb_id != "nan" and self.tmdb_api_key:
            tmdb_data = self._fetch_tmdb_details(tmdb_id)
            if tmdb_data:
                # Poster
                poster_path = tmdb_data.get("poster_path")
                if poster_path:
                    result["poster_url"] = f"{self.tmdb_image_base}{poster_path}"

                # Get TMDB rating (vote_average)
                vote_average = tmdb_data.get("vote_average")
                if vote_average is not None and vote_average > 0:
                    result["tmdb_rating"] = float(vote_average)
                    print(f"[Metadata] TMDB rating for {title}: {result['tmdb_rating']}")

                # Get movie title and year from TMDB for better URL construction
                tmdb_title = tmdb_data.get("title", title)
                tmdb_year = tmdb_data.get("release_date", "")[:4] if tmdb_data.get("release_date") else ""
                
                # Watch providers - when using append_to_response="watch/providers", 
                # it becomes a key "watch/providers" (string key, not nested)
                watch_providers_data = tmdb_data.get("watch/providers", {})
                # Fallback: sometimes it's just "watch"
                if not watch_providers_data:
                    watch_providers_data = tmdb_data.get("watch", {})
                
                # Get US region providers
                if isinstance(watch_providers_data, dict):
                    results_data = watch_providers_data.get("results", {})
                    if isinstance(results_data, dict):
                        us_data = results_data.get("US") or results_data.get("us")
                        
                        if us_data:
                            providers = []
                            # Debug: Log all providers from TMDB
                            print(f"[Metadata] TMDB returned providers for {title}:")
                            for provider_type in ["flatrate", "rent", "buy", "free", "ads"]:
                                provider_list = us_data.get(provider_type, [])
                                if provider_list:
                                    for p in provider_list:
                                        p_name = p.get("provider_name") or p.get("name", "Unknown")
                                        print(f"  - {p_name} ({provider_type})")
                            
                            # Try all provider types
                            for provider_type in ["flatrate", "rent", "buy", "free", "ads"]:
                                provider_list = us_data.get(provider_type, [])
                                if provider_list and isinstance(provider_list, list):
                                    for provider in provider_list:
                                        provider_name = provider.get("provider_name") or provider.get("name")
                                        logo_path = provider.get("logo_path")
                                        provider_id = provider.get("provider_id")  # TMDB provider ID
                                        
                                        if not provider_name:
                                            continue
                                        
                                        # Normalize provider name for matching (lowercase, strip whitespace)
                                        normalized_name = provider_name.strip().lower()
                                        
                                        # Check if provider is in our mapping
                                        display_name = self.provider_mapping.get(normalized_name)
                                        
                                        # If exact match not found, try partial matching
                                        if not display_name:
                                            for tmdb_key, mapped_name in self.provider_mapping.items():
                                                # Check if provider name contains our key or vice versa
                                                if tmdb_key in normalized_name or normalized_name in tmdb_key:
                                                    display_name = mapped_name
                                                    break
                                        
                                        # Skip if not in our allowed providers
                                        if not display_name:
                                            print(f"[Metadata] Skipping provider: {provider_name} (not in allowed list)")
                                            continue
                                        
                                        # Construct direct provider URL
                                        watch_url = self._construct_provider_url(display_name, tmdb_title)
                                        
                                        # Check if we already added this provider (avoid duplicates)
                                        if not any(p["name"] == display_name for p in providers):
                                            print(f"[Metadata] Adding provider: {display_name} for {tmdb_title} (URL: {watch_url})")
                                            providers.append(
                                                {
                                                    "name": display_name,
                                                    "type": provider_type,  # "flatrate" = streaming, "rent" = rent, "buy" = buy
                                                    "logo": f"https://image.tmdb.org/t/p/w92{logo_path}" if logo_path else None,
                                                    "url": watch_url,  # Direct link to provider
                                                    "provider_id": provider_id,
                                                }
                                            )
                            
                            if providers:
                                result["watch_providers"] = providers

        # Always fetch IMDB rating if we have an IMDB ID (regardless of TMDB status)
        if imdb_id and imdb_id.isdigit():
            imdb_rating = self._get_imdb_rating(imdb_id)
            if imdb_rating is not None:
                result["imdb_rating"] = imdb_rating
                print(f"[Metadata] IMDB rating for {title}: {result['imdb_rating']}")
            else:
                print(f"[Metadata] Could not fetch IMDB rating for {title} (IMDB ID: {imdb_id})")

        return result

    def enrich_batch(self, movies: list[Dict]) -> list[Dict]:
        """Enrich a batch of movies with rate limiting to avoid API throttling."""
        enriched = []
        total = len(movies)
        print(f"[Metadata] Enriching {total} movies with TMDB API (rate limited to avoid throttling)...")
        
        for idx, movie in enumerate(movies, 1):
            movie_id = movie.get("movie_id")
            title = movie.get("title", "Unknown")
            
            # Check if we have TMDB ID for this movie
            tmdb_id = self.movie_id_to_tmdb.get(movie_id, "")
            if tmdb_id and tmdb_id != "" and tmdb_id != "nan":
                metadata = self.enrich_movie(movie_id, title)
                movie.update(metadata)
            else:
                # Still try to get IMDB URL if available
                imdb_id = self.movie_id_to_imdb.get(movie_id, "")
                if imdb_id and imdb_id != "" and imdb_id != "nan":
                    if imdb_id.isdigit():
                        padded_id = imdb_id.zfill(7)
                        movie["imdb_url"] = f"https://www.imdb.com/title/tt{padded_id}/"
                        # Try to get IMDB rating (with rate limiting)
                        movie["imdb_rating"] = self._get_imdb_rating(imdb_id)
            
            # Progress logging
            if idx % 3 == 0 or idx == total:
                print(f"[Metadata] Enriched {idx}/{total} movies...")
            
            enriched.append(movie)
        
        print(f"[Metadata] Enrichment complete.")
        return enriched


@lru_cache(maxsize=1)
def get_metadata_service() -> MetadataService:
    """Cached singleton instance."""
    return MetadataService()


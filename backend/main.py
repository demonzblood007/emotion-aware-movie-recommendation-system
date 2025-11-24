from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.dependencies.settings import get_settings
from backend.routers import emotions, genres, ranking, recommendations, users

app = FastAPI(title="Emotion-Aware Movie Recommender API")

# CORS middleware - allow localhost and all ngrok domains
# For production, restrict this to specific domains
# NOTE: Middleware should be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def cors_debug_middleware(request: Request, call_next):
    """Debug middleware - only log errors, not every request."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"[CORS] Error processing {request.method} {request.url.path}: {e}")
        raise

# Granular endpoints
app.include_router(emotions.router)
app.include_router(genres.router)
app.include_router(users.router)
app.include_router(ranking.router)

# Combined endpoint (backward compatibility)
app.include_router(recommendations.router)

@app.get("/health", tags=["system"])
def health_check(settings = Depends(get_settings)) -> dict:
    """Simple health endpoint until other routers are registered."""
    return {"status": "ok", "env": settings.env}

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle CORS preflight requests."""
    return {"status": "ok"}

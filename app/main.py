import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

# Configure logging from environment
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Import routers
from app.routes import health, search, logging, helsedir, content, temaside, roles, dev


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    # --- Startup ---
    print(f"Starting Helsedirektoratet AI Backend in {settings.environment} mode")
    print(f"Database: {settings.mysql_database} @ {settings.mysql_host}:{settings.mysql_port}")
    print(f"ML models directory: {settings.ml_models_dir}")

    # Pre-build BM25 index so first search is fast
    print("\nPre-building BM25 index...")
    from app.services.search.bm25_search import bm25_search
    try:
        count = bm25_search.prebuild()
        print(f"BM25 index built ({count} items)")
    except Exception as e:
        print(f"Failed to pre-build BM25 index: {e}")
        print("  BM25 index will be built lazily on first search")

    # Pre-load embeddings if enabled
    if settings.ml_embedding_enabled:
        print("\nPre-loading embeddings...")
        from app.services.search.semantic_search import semantic_search

        try:
            # Load model
            if semantic_search.load_embedding_model():
                # Force load the actual transformer model (not just initialize)
                if semantic_search.embedding_model:
                    semantic_search.embedding_model._load_model()
                print("Embedding model loaded")
            else:
                print("Failed to load embedding model")

            # Load embeddings into cache
            if semantic_search.load_content_embeddings():
                print(f"Loaded {len(semantic_search.content_embeddings)} embeddings into cache")
            else:
                print("Failed to load embeddings")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("  App will start without semantic search")
    else:
        print("\nEmbedding search disabled (ML_EMBEDDING_ENABLED=false)")

    # Pre-load ranking model if enabled
    if settings.ml_ranking_enabled:
        print("\nPre-loading ranking model...")
        from app.services.search.ml_service import ml_service
        if ml_service.load_ranking_model():
            print("Ranking model loaded")
        else:
            print("Failed to load ranking model (will retry on first search)")
    else:
        print("\nRanking model disabled (ML_RANKING_ENABLED=false)")

    yield

    # --- Shutdown ---
    print("Shutting down Helsedirektoratet AI Backend")


# Create FastAPI app
app = FastAPI(
    title="Helsedirektoratet AI Backend",
    description="Backend API for AI-powered content search and recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(search.router)
app.include_router(content.router)
app.include_router(logging.router)
app.include_router(helsedir.router)
app.include_router(temaside.router)
app.include_router(roles.router)
app.include_router(dev.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment == "development"),
    )

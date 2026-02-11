from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

# Import routers
from app.routes import health, search, logging, helsedir, content, temaside

# Create FastAPI app
app = FastAPI(
    title="Helsedirektoratet AI Backend",
    description="Backend API for AI-powered content search and recommendations",
    version="1.0.0",
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


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    print(f"Starting Helsedirektoratet AI Backend in {settings.environment} mode")
    print(f"Database: {settings.mysql_database} @ {settings.mysql_host}:{settings.mysql_port}")
    print(f"ML models directory: {settings.ml_models_dir}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run shutdown tasks."""
    print("Shutting down Helsedirektoratet AI Backend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment == "development"),
    )

"""
Simple script to run the FastAPI server with loaded configuration.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import app
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("=" * 50)
    print("Starting Helsedirektoratet AI Backend")
    print("=" * 50)
    print(f"Environment: {settings.environment}")
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Database: {settings.mysql_database} @ {settings.mysql_host}:{settings.mysql_port}")
    print("-" * 50)
    print(f"API Documentation:")
    print(f"  - Swagger UI: http://localhost:{settings.port}/docs")
    print(f"  - ReDoc: http://localhost:{settings.port}/redoc")
    print("=" * 50)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.environment == "development"),
        log_level="info",
    )

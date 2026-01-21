from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Priority order:
    1. System environment variables
    2. .env file values
    3. Default values defined below
    """

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"
    debug: bool = False

    # Data paths
    content_file: str = "data/content.json"
    logs_file: str = "logs/logs.jsonl"

    # Helsedirektoratet API
    helsedir_api_key: str = ""
    helsedir_api_url: str = "https://api.helsedirektoratet.no"

    # Optional: OpenAI
    openai_api_key: str = ""

    # Search scoring weights
    search_exact_phrase_title_weight: float = 10.0
    search_keyword_title_weight: float = 3.0
    search_keyword_body_weight: float = 1.0
    search_exact_phrase_body_weight: float = 2.0
    search_tag_match_weight: float = 2.0

    # ML settings
    ml_embedding_enabled: bool = False
    ml_ranking_enabled: bool = False
    ml_models_dir: str = "models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Ensure directories exist
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)
Path(settings.ml_models_dir).mkdir(exist_ok=True)

# Print loaded configuration (useful for debugging)
if settings.debug:
    print(f"Configuration loaded:")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Environment: {settings.environment}")
    print(f"  Content file: {settings.content_file}")

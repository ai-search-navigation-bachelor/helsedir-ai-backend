from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Priority order:
    1. System environment variables
    2. .env file values
    3. Default values defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"
    debug: bool = False

    # Helsedirektoratet API
    helsedir_api_key: str = ""
    helsedir_api_url: str = "https://api.helsedirektoratet.no"

    # Search method: 'keyword', 'semantic', or 'hybrid'
    search_method: str = "hybrid"

    # Search scoring weights (title-only)
    search_exact_phrase_title_weight: float = 10.0
    search_full_title_coverage_weight: float = 7.0  # All title words in query
    search_keyword_title_weight: float = 3.0

    # Categorized search settings
    search_min_score: float = 0.4  # Minimum score threshold for results
    search_category_preview_count: int = 5  # Number of results in category preview

    # ML settings
    ml_embedding_enabled: bool = False
    ml_ranking_enabled: bool = False
    ml_models_dir: str = "models"

    # MySQL Database
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "helsedir_ai_user"
    mysql_password: str = "your_password_here"
    mysql_database: str = "helsedir_ai"


# Create settings instance
settings = Settings()

# Ensure ML models directory exists
Path(settings.ml_models_dir).mkdir(parents=True, exist_ok=True)

# Print loaded configuration (useful for debugging)
if settings.debug:
    print(f"Configuration loaded:")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Environment: {settings.environment}")
    print(f"  Database: {settings.mysql_database}")
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Ensure directories exist
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Print loaded configuration (useful for debugging)
if settings.debug:
    print(f"Configuration loaded:")
    print(f"  Host: {settings.host}")
    print(f"  Port: {settings.port}")
    print(f"  Environment: {settings.environment}")
    print(f"  Content file: {settings.content_file}")

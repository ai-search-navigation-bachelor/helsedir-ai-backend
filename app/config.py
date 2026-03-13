from pydantic import Field
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
    log_level: str = "WARNING"

    # Helsedirektoratet API
    helsedir_api_key: str = ""
    helsedir_api_url: str = "https://api.helsedirektoratet.no"
    helsedir_public_base_url: str = "https://www.helsedirektoratet.no"

    # Search method: 'keyword', 'semantic', or 'hybrid'
    search_method: str = "hybrid"

    # Search scoring weights (title-only)
    search_exact_phrase_title_weight: float = 10.0
    search_full_title_coverage_weight: float = 7.0  # All title words in query
    search_keyword_title_weight: float = 3.0

    # Hybrid search settings
    search_rrf_k: int = 60  # RRF fusion constant (higher = more weight to top ranks)
    search_rrf_weight_bm25: float = 0.3  # RRF weight for BM25 retrieval
    search_rrf_weight_semantic: float = 0.7  # RRF weight for semantic retrieval
    search_bm25_hierarchy_enabled: bool = True
    search_bm25_hierarchy_max_depth: int = 4
    search_bm25_hierarchy_decay: float = 0.65
    search_bm25_hierarchy_source_top_k: int = 400
    search_bm25_hierarchy_top_children: int = 3
    search_bm25_hierarchy_tail_weight: float = 0.35
    search_bm25_hierarchy_weight: float = 0.8
    search_bm25_hierarchy_min_contribution: float = 0.0001
    search_bm25_title_weight: int = 3

    # Content type score boosts (multiplied on combined_score after RRF fusion)
    search_boost_temaside: float = 1.15
    search_boost_retningslinje: float = 1.10

    # Role match boost/penalty (multiplied on combined_score in hybrid search)
    search_role_match_boost: float = 1.15
    search_role_mismatch_penalty: float = 0.85

    # Categorized search settings
    search_min_score: float = 0.45  # Minimum score threshold for results
    search_category_preview_count: int = 3  # Number of results in category preview

    # ML settings
    ml_embedding_enabled: bool = False
    ml_embedding_model: str = "models/finetuned-e5-gpl"  # Path to finetuned model
    ml_ranking_enabled: bool = False
    ml_models_dir: str = "models"

    # LLM API
    openai_api_key: str = ""
    # Groq — up to 6 keys for parallel generation
    groq_api_key: str = ""
    groq_api_key_2: str = ""
    groq_api_key_3: str = ""
    groq_api_key_4: str = ""
    groq_api_key_5: str = ""
    groq_api_key_6: str = ""

    @property
    def groq_api_keys(self) -> list[str]:
        """Return all configured Groq API keys (non-empty)."""
        return [
            k for k in [
                self.groq_api_key,
                self.groq_api_key_2,
                self.groq_api_key_3,
                self.groq_api_key_4,
                self.groq_api_key_5,
                self.groq_api_key_6,
            ]
            if k
        ]

    # MySQL Database
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3307
    mysql_user: str = "helsedir_ai_user"
    mysql_password: str = "your_password_here"
    mysql_database: str = "helsedir_ai"
    # Per-worker pool size; multiply by worker count when sizing against max_connections
    mysql_pool_size: int = Field(default=10, le=32)
    mysql_root_password: str = ""


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

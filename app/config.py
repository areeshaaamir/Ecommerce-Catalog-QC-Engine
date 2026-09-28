import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Configuration
    APP_NAME: str = "E-Commerce Catalog QC Engine"
    DEBUG: bool = True
    DATABASE_URL: str = "catalog_qc.db"
    DATABASE_PATH: str = "catalog_qc.db"

    # OpenRouter Configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openrouter/free"
    VISION_MODEL: str = "openrouter/free"

    # Vision & Embedding Settings
    VISION_CONFIDENCE_THRESHOLD: float = 0.70
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"

    # Similarity Threshold for Matching
    SIMILARITY_THRESHOLD: float = 0.25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "catalog_qc.db")
    VISION_MODEL: str = os.getenv("VISION_MODEL", "gemini-3.8-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    
    VISION_CONFIDENCE_THRESHOLD: float = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.80"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

settings = Settings()

if not settings.GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set in environment or .env file!")
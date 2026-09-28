import os
import math
from openai import OpenAI
from app.config import settings

api_key = getattr(settings, "OPENROUTER_API_KEY", None) or os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key if api_key else "placeholder"
)

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def generate_text_embedding(text: str) -> list[float]:
    """Generates vector embeddings using OpenRouter's embeddings API."""
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model_name = getattr(settings, "EMBEDDING_MODEL", "openai/text-embedding-3-small")

    response = client.embeddings.create(
        model=model_name,
        input=text
    )

    return response.data[0].embedding
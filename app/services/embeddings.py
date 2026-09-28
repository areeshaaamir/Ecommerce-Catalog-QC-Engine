import math
from google import genai
from app.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None

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
    """Generates a text vector embedding using active Gemini embedding endpoints."""
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    # Candidate embedding model endpoints
    candidate_models = ["gemini-embedding-2", "gemini-embedding-001", "text-embedding-004"]
    result = None
    last_error = None

    for model in candidate_models:
        try:
            result = client.models.embed_content(
                model=model,
                contents=text,
            )
            break
        except Exception as e:
            last_error = e
            continue

    if not result:
        raise RuntimeError(f"All embedding models failed. Last error: {last_error}")

    # Extract float values from response object
    if hasattr(result, "embedding") and hasattr(result.embedding, "values"):
        return result.embedding.values
    elif hasattr(result, "embeddings") and len(result.embeddings) > 0:
        return result.embeddings[0].values
    else:
        raise ValueError("Unexpected response format from embedding API.")
import sqlite3
from app.config import settings
from app.db import get_db_connection
from app.services.embeddings import cosine_similarity, generate_text_embedding


def verify_catalog_listing(photo_id: str, listing_title: str, listing_description: str = "") -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT p.status, t.subject, t.category, t.color, t.material, t.caption
        FROM photos p
        LEFT JOIN photo_tags t ON p.id = t.photo_id
        WHERE p.id = ?
        """,
        (photo_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {"error": f"Photo ID '{photo_id}' not found in database."}

    if row["status"] != "PROCESSED":
        return {"error": f"Photo ID '{photo_id}' is not in PROCESSED state (Current status: {row['status']})."}

    # Concatenate structured tags with clean fallback logic
    subject = row['subject'] or ''
    category = row['category'] or ''
    color = row['color'] or ''
    material = row['material'] or ''
    caption = row['caption'] or ''

    # Concise context payload gives higher semantic similarity scores to titles
    photo_context = f"Product: {color} {material} {subject} ({category}). Description: {caption}"
    listing_full_text = f"{listing_title}. {listing_description}".strip()

    # Generate embeddings via OpenRouter
    photo_embedding = generate_text_embedding(photo_context)
    listing_embedding = generate_text_embedding(listing_full_text)

    similarity_score = cosine_similarity(photo_embedding, listing_embedding)

    threshold = getattr(settings, "SIMILARITY_THRESHOLD", 0.25)
    is_match = similarity_score >= threshold

    return {
        "photo_id": photo_id,
        "photo_summary": photo_context,
        "listing_text": listing_full_text,
        "similarity_score": round(similarity_score, 4),
        "threshold": threshold,
        "verdict": "MATCH" if is_match else "MISMATCH",
        "is_flagged": not is_match
    }
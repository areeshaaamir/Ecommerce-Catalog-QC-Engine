import json
import sqlite3
import uuid

from app.config import settings
from app.db import get_db_connection
from app.services.vision import analyze_product_image


def log_api_cost(conn: sqlite3.Connection, job_type: str, model_name: str, metrics: dict):
    """Logs token consumption and API cost estimates into cost_logs table."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO cost_logs (id, job_type, model_name, prompt_tokens, completion_tokens, estimated_cost_usd)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            job_type,
            model_name,
            metrics.get("prompt_tokens", 0),
            metrics.get("completion_tokens", 0),
            metrics.get("estimated_cost_usd", 0.0),
        ),
    )


def process_photo_ingestion(photo_id: str, file_path: str):
    """
    Background execution workflow:
    1. Analyzes image using analyze_product_image().
    2. Logs token costs to SQLite.
    3. Evaluates vision confidence score against threshold.
    4. Saves extracted tags into photo_tags table.
    5. Updates photos table status to 'PROCESSED', 'FLAGGED', or 'FAILED'.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Step 1: Run image through Gemini Vision
        extracted_tags, metrics = analyze_product_image(file_path)

        # Step 2: Log token costs
        log_api_cost(conn, "VISION_TAGGING", settings.VISION_MODEL, metrics)

        # Step 3: Flag low-confidence output
        status = "PROCESSED"
        if extracted_tags.confidence < settings.VISION_CONFIDENCE_THRESHOLD:
            status = "FLAGGED"

        # Step 4: Insert extracted tags into photo_tags
        tag_id = str(uuid.uuid4())
        cursor.execute(
            """
            INSERT OR REPLACE INTO photo_tags 
            (id, photo_id, subject, category, color, material, attributes_json, caption, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tag_id,
                photo_id,
                extracted_tags.subject.lower(),
                extracted_tags.category.lower(),
                extracted_tags.color.lower(),
                extracted_tags.material.lower(),
                json.dumps(extracted_tags.attributes),  # Convert list to JSON string for SQLite
                extracted_tags.caption,
                extracted_tags.confidence,
            ),
        )

        # Step 5: Update main photo status
        cursor.execute("UPDATE photos SET status = ? WHERE id = ?", (status, photo_id))
        conn.commit()
        print(f"Success: {file_path} marked as {status} (Confidence: {extracted_tags.confidence})")

    except Exception as e:
        conn.rollback()
        cursor.execute("UPDATE photos SET status = 'FAILED' WHERE id = ?", (photo_id,))
        conn.commit()
        print(f"Error processing {file_path}: {str(e)}")

    finally:
        conn.close()
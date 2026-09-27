import json
import uuid
from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from app.config import settings
from app.db import get_db_connection, init_db
from app.schema import PhotoIngestRequest, PhotoIngestResponse
from app.services.pipeline import process_photo_ingestion

app = FastAPI(
    title="E-Commerce Catalog QC Engine",
    description="AI-Powered Product Photo & Listing Mismatch Guard",
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": settings.DATABASE_PATH,
        "vision_model": settings.VISION_MODEL,
    }


@app.post(
    "/api/v1/photos/ingest",
    response_model=PhotoIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_photo(payload: PhotoIngestRequest, background_tasks: BackgroundTasks):
    """Ingests a local image path and queues background vision extraction."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if photo was already registered
    cursor.execute("SELECT id, status FROM photos WHERE file_path = ?", (payload.file_path,))
    existing = cursor.fetchone()

    if existing:
        conn.close()
        return PhotoIngestResponse(
            photo_id=existing["id"],
            file_path=payload.file_path,
            status=existing["status"],
            reason="Photo already registered",
        )

    # Register new photo record with PENDING status
    photo_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO photos (id, file_path, status) VALUES (?, ?, ?)",
        (photo_id, payload.file_path, "PENDING"),
    )
    conn.commit()
    conn.close()

    # Trigger background pipeline execution
    background_tasks.add_task(process_photo_ingestion, photo_id, payload.file_path)

    return PhotoIngestResponse(
        photo_id=photo_id,
        file_path=payload.file_path,
        status="PENDING",
        reason="Vision processing queued in background",
    )


@app.get("/api/v1/photos/{photo_id}")
def get_photo_status(photo_id: str):
    """Retrieves status and structured tags for a given photo ID."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT p.id, p.file_path, p.status, p.created_at,
               t.subject, t.category, t.color, t.material, 
               t.attributes_json, t.caption, t.confidence
        FROM photos p
        LEFT JOIN photo_tags t ON p.id = t.photo_id
        WHERE p.id = ?
        """,
        (photo_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Photo ID not found")

    result = dict(row)
    if result["attributes_json"]:
        result["attributes"] = json.loads(result["attributes_json"])
        del result["attributes_json"]

    return result
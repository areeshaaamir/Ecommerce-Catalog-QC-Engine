import json
import os
import shutil
import uuid
from typing import List, Optional
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import settings
from app.db import get_db_connection, init_db
from app.schema import PhotoIngestRequest, PhotoIngestResponse
from app.services.pipeline import process_photo_ingestion
from app.services.qc_engine import verify_catalog_listing
from app.services.embeddings import generate_text_embedding, cosine_similarity

app = FastAPI(
    title="E-Commerce Catalog QC Engine",
    description="AI-Powered Product Photo & Listing Mismatch Guard",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Directory where catalog images are stored
IMAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "image data"))
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/image-data", StaticFiles(directory=IMAGE_DIR), name="image-data")


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


# ==========================================
# NEW: DIRECT IMAGE UPLOAD & AUTO-PROCESSING
# ==========================================
@app.post("/dashboard/upload")
async def upload_catalog_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Saves an uploaded image file, registers it in the DB, 
    and triggers background vision tagging & embedding extraction.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    # Clean filename and build destination path
    safe_filename = os.path.basename(file.filename)
    dest_path = os.path.join(IMAGE_DIR, safe_filename)

    # Save file to disk
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn = get_db_connection()
    cursor = conn.cursor()

    # Relative path stored in DB to maintain consistency
    db_file_path = os.path.join("image data", safe_filename)

    # Check if image path already exists
    cursor.execute("SELECT id FROM photos WHERE file_path = ?", (db_file_path,))
    existing = cursor.fetchone()

    if existing:
        photo_id = existing["id"]
        cursor.execute("UPDATE photos SET status = 'PENDING' WHERE id = ?", (photo_id,))
    else:
        photo_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO photos (id, file_path, status, created_at) VALUES (?, ?, 'PENDING', datetime('now'))",
            (photo_id, db_file_path),
        )

    conn.commit()
    conn.close()

    # Trigger async AI processing pipeline
    background_tasks.add_task(process_photo_ingestion, photo_id, db_file_path)

    # Redirect back to dashboard gallery
    return RedirectResponse(url="/dashboard#gallery", status_code=303)


# ==========================================
# REAL-TIME ADMIN DASHBOARD + REVERSE SEARCH
# ==========================================
@app.get("/dashboard")
def view_qc_dashboard(request: Request, q: Optional[str] = None):
    """
    Renders real-time Quality Control Audit Dashboard with reverse search.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Reverse Semantic Image Search
    search_results = []
    SEARCH_RELEVANCE_THRESHOLD = 0.35

    if q and q.strip():
        search_query = q.strip()
        cursor.execute("""
            SELECT p.id, p.file_path, t.subject, t.category, t.color, t.material, t.caption
            FROM photos p
            JOIN photo_tags t ON p.id = t.photo_id
            WHERE p.status = 'PROCESSED'
        """)
        catalog_rows = cursor.fetchall()

        if catalog_rows:
            query_embedding = generate_text_embedding(search_query)
            scored_matches = []

            for row in catalog_rows:
                photo_context = (
                    f"Product: {row['color']} {row['material']} {row['subject']} ({row['category']}). "
                    f"Description: {row['caption']}"
                )
                photo_embedding = generate_text_embedding(photo_context)
                score = cosine_similarity(query_embedding, photo_embedding)

                if score >= SEARCH_RELEVANCE_THRESHOLD:
                    relative_file = os.path.basename(row['file_path'])
                    image_url = f"/image-data/{relative_file}"

                    scored_matches.append({
                        "photo_id": row["id"],
                        "file_path": row["file_path"],
                        "image_url": image_url,
                        "score": score,
                        "context": photo_context,
                        "category": row["category"],
                        "color": row["color"],
                        "subject": row["subject"]
                    })

            scored_matches.sort(key=lambda x: x["score"], reverse=True)
            search_results = scored_matches[:3]

    # 2. Retrieve All Ingested Catalog Images
    cursor.execute("""
        SELECT p.id, p.file_path, p.status, t.subject, t.category, t.color, t.material, t.caption
        FROM photos p
        LEFT JOIN photo_tags t ON p.id = t.photo_id
        ORDER BY p.created_at DESC
    """)
    all_images = []
    for r in cursor.fetchall():
        relative_file = os.path.basename(r["file_path"])
        all_images.append({
            "id": r["id"],
            "file_path": r["file_path"],
            "image_url": f"/image-data/{relative_file}",
            "status": r["status"],
            "category": r["category"] or "N/A",
            "subject": r["subject"] or "N/A",
            "caption": r["caption"] or "Processing AI Vision..."
        })

    # 3. Retrieve Audit Log
    try:
        cursor.execute("""
            SELECT 
                e.id, e.photo_id, e.listing_id, e.similarity_score, 
                e.status AS verdict, e.rejection_reason, e.evaluated_at,
                t.caption, t.subject, t.category, t.color, t.material
            FROM evaluations e
            LEFT JOIN photo_tags t ON e.photo_id = t.photo_id
            ORDER BY e.id DESC
        """)
        raw_rows = [dict(row) for row in cursor.fetchall()]

        evaluations = []
        for row in raw_rows:
            color = row.get("color") or ""
            material = row.get("material") or ""
            subject = row.get("subject") or "product"
            category = row.get("category") or "general"
            caption = row.get("caption") or ""
            
            photo_summary = f"{color} {material} {subject} ({category}) - {caption}".strip(" -")
            if not photo_summary or photo_summary == "()":
                photo_summary = "Visual tags processing..."

            evaluations.append({
                "photo_id": row.get("photo_id", "N/A"),
                "listing_text": f"Listing #{row.get('listing_id', 'N/A')}",
                "photo_summary": photo_summary,
                "similarity_score": float(row.get("similarity_score") or 0.0),
                "verdict": row.get("verdict", "UNKNOWN").upper(),
                "threshold": getattr(settings, "SIMILARITY_THRESHOLD", 0.30)
            })
    except Exception as e:
        print(f"[DASHBOARD ERROR] Failed to query evaluations table: {e}")
        evaluations = []

    total = len(evaluations)
    matches = sum(1 for e in evaluations if e["verdict"] == "MATCH")
    mismatches = sum(1 for e in evaluations if e["verdict"] == "MISMATCH")
    avg_score = round(sum(e["similarity_score"] for e in evaluations) / total, 4) if total > 0 else 0.0000

    conn.close()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "query": q or "",
            "search_results": search_results,
            "all_images": all_images,
            "total_evaluations": total,
            "matches_count": matches,
            "mismatches_count": mismatches,
            "avg_score": avg_score,
            "evaluations": evaluations,
        }
    )


# ==========================================
# REST API ENDPOINTS
# ==========================================
@app.post("/api/v1/photos/ingest", status_code=202)
def ingest_photo(payload: PhotoIngestRequest, background_tasks: BackgroundTasks):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, status FROM photos WHERE file_path = ?", (payload.file_path,))
    existing = cursor.fetchone()

    if existing:
        photo_id = existing["id"]
        status_val = existing["status"]

        if status_val == "PROCESSED":
            conn.close()
            return {
                "photo_id": photo_id,
                "file_path": payload.file_path,
                "status": "PROCESSED",
                "message": "Photo is already processed. Skipping re-processing."
            }

        cursor.execute("UPDATE photos SET status = 'PENDING' WHERE id = ?", (photo_id,))
        conn.commit()
        conn.close()

        background_tasks.add_task(process_photo_ingestion, photo_id, payload.file_path)
        return {
            "photo_id": photo_id,
            "file_path": payload.file_path,
            "status": "PENDING",
            "message": "Photo previously failed/pending. Re-queued for processing."
        }

    photo_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO photos (id, file_path, status, created_at) VALUES (?, ?, 'PENDING', datetime('now'))",
        (photo_id, payload.file_path),
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(process_photo_ingestion, photo_id, payload.file_path)

    return {
        "photo_id": photo_id,
        "file_path": payload.file_path,
        "status": "PENDING",
        "message": "Photo queued for processing."
    }


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


class QCVerifyRequest(BaseModel):
    photo_id: str
    listing_title: str
    listing_description: Optional[str] = ""


@app.post("/api/v1/qc/verify")
def verify_listing_endpoint(payload: QCVerifyRequest):
    """
    Quality Control endpoint comparing a product listing to a photo.
    """
    result = verify_catalog_listing(
        photo_id=payload.photo_id,
        listing_title=payload.listing_title,
        listing_description=payload.listing_description,
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
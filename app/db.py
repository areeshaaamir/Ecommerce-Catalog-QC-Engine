import sqlite3
import os
from app.config import settings

def get_db_connection():
    """Returns a connection to SQLite database with Row factory enabled."""
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes all database tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        sku TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS listings (
        id TEXT PRIMARY KEY,
        product_id TEXT NOT NULL,
        sku TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        expected_category TEXT NOT NULL,
        expected_color TEXT,
        expected_material TEXT,
        embedding_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS photos (
        id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL DEFAULT 'PENDING', -- PENDING, PROCESSED, FLAGGED, FAILED
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS photo_tags (
        id TEXT PRIMARY KEY,
        photo_id TEXT NOT NULL UNIQUE,
        subject TEXT NOT NULL,
        category TEXT NOT NULL,
        color TEXT NOT NULL,
        material TEXT NOT NULL,
        attributes_json TEXT NOT NULL,
        caption TEXT NOT NULL,
        confidence REAL NOT NULL,
        embedding_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,
        listing_id TEXT NOT NULL,
        photo_id TEXT NOT NULL,
        similarity_score REAL NOT NULL,
        status TEXT NOT NULL, -- APPROVED, REJECTED, FLAGGED
        rejection_reason TEXT,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(listing_id) REFERENCES listings(id) ON DELETE CASCADE,
        FOREIGN KEY(photo_id) REFERENCES photos(id) ON DELETE CASCADE
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cost_logs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL, -- VISION_TAGGING, EMBEDDING
        model_name TEXT NOT NULL,
        prompt_tokens INTEGER DEFAULT 0,
        completion_tokens INTEGER DEFAULT 0,
        estimated_cost_usd REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_listings_sku ON listings(sku);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(status);")

    conn.commit()
    conn.close()
    print("Database tables initialized successfully.")

if __name__ == "__main__":
    init_db()
# AI-Powered E-Commerce Catalog Quality Control & Reverse Search Engine

An automated end-to-end **Quality Control (QC) engine for e-commerce platforms**.

Built with **FastAPI, SQLite, and OpenAI**, this system ingests product images, extracts structured visual attributes using GPT Vision, generates vector embeddings, and performs semantic reverse image lookup to detect mismatches between product photos and catalog listing titles/descriptions.

---

## Key Features

### Vision Tagging Pipeline

Automatically processes catalog images using GPT Vision to extract structured attributes:

* Category
* Color
* Material
* Subject
* Detailed caption

### Semantic Reverse Search

Uses vector similarity with OpenAI's `text-embedding-3-small` model to search product images using natural-language queries.

A configurable relevance threshold is used to suppress false positives.

### Catalog Mismatch Guard

Compares listing metadata against AI-extracted image attributes using cosine similarity to flag potential:

* Mislabeled products
* Incorrect product images
* Catalog metadata mismatches

### Real-Time Web Dashboard

A responsive admin panel built with **Jinja2** and **Tailwind CSS**, featuring:

* Audit log statistics
* Search candidate rankings
* Catalog image gallery
* Image repository controls
* Reverse image search

### Live File Upload

Upload product images directly through the web interface.

Image processing is executed asynchronously using FastAPI's `BackgroundTasks`.

---

## Project Structure

```text
.
├── app/
│   ├── config.py
│   ├── db.py
│   ├── main.py
│   ├── schema.py
│   │
│   ├── services/
│   │   ├── embeddings.py
│   │   ├── pipeline.py
│   │   ├── qc_engine.py
│   │   └── vision.py
│   │
│   └── templates/
│       └── dashboard.html
│
├── image data/
├── catalog_qc.db
├── requirements.txt
└── test_lookup.py
```

### File Overview

| File / Directory               | Description                                             |
| ------------------------------ | ------------------------------------------------------- |
| `app/config.py`                | Environment configuration and similarity thresholds     |
| `app/db.py`                    | SQLite database setup and connection helpers            |
| `app/main.py`                  | FastAPI routes and application entry point              |
| `app/schema.py`                | Pydantic request and response schemas                   |
| `app/services/embeddings.py`   | Embedding generation and cosine similarity calculations |
| `app/services/pipeline.py`     | Asynchronous background image-processing workflow       |
| `app/services/qc_engine.py`    | Catalog verification and comparison logic               |
| `app/services/vision.py`       | GPT Vision analysis and JSON tag parsing                |
| `app/templates/dashboard.html` | Tailwind CSS web admin interface                        |
| `image data/`                  | Catalog image storage directory                         |
| `catalog_qc.db`                | Local SQLite database                                   |
| `test_lookup.py`               | CLI test script for semantic lookup                     |

---

## Tech Stack

| Category             | Technologies                                    |
| -------------------- | ----------------------------------------------- |
| **Backend**          | Python 3.10+, FastAPI, Uvicorn                  |
| **Database**         | SQLite                                          |
| **AI / ML**          | OpenAI GPT Vision API, `text-embedding-3-small` |
| **Frontend**         | Jinja2 Templates, Tailwind CSS                  |
| **Image Processing** | PIL / Pillow, `pillow-avif-plugin`              |

---

## Getting Started

### Prerequisites

Make sure you have:

* Python **3.10 or higher**
* An OpenAI API key

---

### 1. Clone the Repository

```bash
git clone https://github.com/areeshaaamir/ecommerce-catalog-qc.git
cd ecommerce-catalog-qc
```

### 2. Create a Virtual Environment

```bash
python -m venv env
```

#### Windows PowerShell

```powershell
.\env\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your-openai-api-key-here
DATABASE_PATH=catalog_qc.db
VISION_MODEL=gpt-4o-mini
SIMILARITY_THRESHOLD=0.30
```

> **Important:** Never commit your `.env` file or expose your OpenAI API key publicly.

Add the following to `.gitignore` if it isn't already present:

```gitignore
.env
env/
__pycache__/
*.pyc
```

---

## ▶Running the Application

Start the FastAPI development server using Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

Once the server is running, open:

| Interface                  | URL                               |
| -------------------------- | --------------------------------- |
| **Web Admin Dashboard** | `http://127.0.0.1:8000/dashboard` |
| **Swagger API Docs**    | `http://127.0.0.1:8000/docs`      |
| **Health Check**        | `http://127.0.0.1:8000/health`    |

---

## Core API Endpoints

| Method | Endpoint                    | Description                                                               |
| ------ | --------------------------- | ------------------------------------------------------------------------- |
| `GET`  | `/dashboard`                | Renders the admin dashboard, catalog gallery, and reverse search          |
| `POST` | `/dashboard/upload`         | Ingests a new image file and triggers asynchronous AI processing          |
| `POST` | `/api/v1/photos/ingest`     | Queues a local photo path for processing via API                          |
| `GET`  | `/api/v1/photos/{photo_id}` | Retrieves AI vision tags and processing status for a photo                |
| `POST` | `/api/v1/qc/verify`         | Performs match verification between a photo and listing title/description |

---

## Testing

Run the CLI lookup test script to interactively verify reverse-search rankings:

```bash
python test_lookup.py
```

---

## Processing Pipeline

The overall workflow can be summarized as:

```text
Product Image
      │
      ▼
┌─────────────────┐
│   Image Upload  │
└────────┬────────┘
         ▼
┌─────────────────┐
│   GPT Vision    │
│  Image Analysis │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Structured Tags │
│ Category/Color  │
│ Material/Etc.   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Text Embeddings │
│ text-embedding- │
│    3-small      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Vector Similarity│
│    Search       │
└────────┬────────┘
         ▼
┌─────────────────┐
│ Catalog QC      │
│ Verification    │
└────────┬────────┘
         ▼
   Match / Mismatch
```

---

## Configuration

The application's behavior can be adjusted through environment variables.

| Variable               | Description                                   | Example         |
| ---------------------- | --------------------------------------------- | --------------- |
| `OPENAI_API_KEY`       | OpenAI API authentication key                 | `your-api-key`  |
| `DATABASE_PATH`        | SQLite database location                      | `catalog_qc.db` |
| `VISION_MODEL`         | OpenAI vision model used for image analysis   | `gpt-4o-mini`   |
| `SIMILARITY_THRESHOLD` | Minimum similarity score for relevant results | `0.30`          |

## Author

**Areesha Aamir**


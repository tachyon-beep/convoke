# Production Web Interface Plan for CrewAI Orchestration System

## 1. Architecture Overview

- **Backend:** FastAPI (Python, async, robust, easy to serve files and APIs)
- **Task Queue:** Celery (with Redis or RabbitMQ) for async, long-running orchestration jobs
- **Frontend:** React (or Vue/Svelte) SPA, or Streamlit for rapid prototyping
- **File Storage:** Local filesystem (for MVP), S3 or similar for scaling
- **Database:** SQLite/PostgreSQL for job metadata, status, and user management (optional for MVP)
- **Authentication:** JWT or OAuth2 (optional for MVP)

---

## 2. Backend (FastAPI) Specification

### Endpoints

- `POST /jobs/`  
  Submit a new orchestration job. Input: requirements/config. Output: Job ID, status.
- `GET /jobs/{job_id}`  
  Get job status and summary.
- `GET /jobs/{job_id}/result`  
  Download or view the final result (JSON, tree, code, etc.)
- `GET /jobs/{job_id}/artifact/{path:path}`  
  Download any generated file (code, review, test, etc.)
- `GET /jobs/`  
  List all jobs (with filters for user, status, etc.)
- `GET /logs/{job_id}`  
  Stream or fetch logs for a job (for real-time feedback)

### Async Orchestration

- On `POST /jobs/`, enqueue a Celery task to run `orchestrate_full_workflow`.
- Store job metadata and status in a database (or in Redis for MVP).
- Update job status as the workflow progresses (pending, running, completed, failed).
- Save all output files to a per-job directory (e.g., `output/{job_id}/...`).

### File Serving

- Serve generated files via `/jobs/{job_id}/artifact/{path}`.
- Optionally, provide zipped download of the entire job output.

### Security

- (Optional) Require authentication for job submission and file access.

---

## 3. Frontend Specification

- **Job Submission Form:**
  - Text area for requirements
  - Config options (max depth, items, etc.)
  - Submit button
- **Job List & Status Page:**
  - List of jobs with status/progress
  - Click to view details
- **Job Detail Page:**
  - Progress bar or live log stream
  - Download links for artifacts (code, reviews, tree, etc.)
  - Inline code/review/test previews
- **Visualization:**
  - Render PROJECT_TREE.txt as a collapsible tree
  - Show code and test files in syntax-highlighted viewers
- **Notifications:**
  - Toasts or banners for job completion/failure

---

## 4. Async Processing & Scaling

- **Celery Worker(s):**
  - Run orchestration jobs in the background
  - Can scale horizontally for more throughput
- **Redis/RabbitMQ:**
  - Message broker for Celery
- **Database:**
  - Store job metadata, status, and (optionally) user info

---

## 5. Deployment

- **Backend:**
  - Run FastAPI with Uvicorn/Gunicorn
  - Run Celery worker(s) in separate process/container
- **Frontend:**
  - Serve static files via Nginx or from FastAPI
  - Or deploy as a separate service (e.g., Vercel, Netlify)
- **File Storage:**
  - Local disk for MVP, S3 for production
- **Environment:**
  - Use Docker Compose for local dev and deployment

---

## 6. MVP Implementation Steps

1. **Backend**
   - [ ] Refactor `orchestrate_full_workflow` to be callable as a Celery task
   - [ ] Implement FastAPI endpoints for job submission, status, and file serving
   - [ ] Store job outputs in `output/{job_id}/`
   - [ ] Add job status tracking (in-memory, Redis, or DB)
2. **Frontend**
   - [ ] Build a simple React (or Streamlit) UI for job submission and status
   - [ ] Add file download and code preview
3. **Async**
   - [ ] Set up Celery with Redis
   - [ ] Ensure jobs run in background and status is updated
4. **(Optional) Auth & User Management**
   - [ ] Add JWT/OAuth2 for user login and job ownership

---

## 7. Example Directory Structure

```
convoke/
    basic.py
    api/                # FastAPI app
        main.py
        celery_worker.py
        models.py
        ...
    output/
        {job_id}/
            architecture.json
            module_X/
                ...
    frontend/
        ...            # React or Streamlit app
    requirements.txt
    docker-compose.yml
```

---

## 8. Tech Stack Summary

- **Backend:** FastAPI, Celery, Redis, Python
- **Frontend:** React (or Streamlit for MVP)
- **Storage:** Local FS (MVP), S3 (prod)
- **Deployment:** Docker Compose, Nginx

---

**This plan provides a robust, scalable foundation for a production web interface for your CrewAI orchestration system.**

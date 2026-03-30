"""
FastAPI Application for Job Data
Serves scraped job data with filtering and pagination
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import json
import os
from typing import Optional, List
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Job Scraper API",
    description="API to access scraped remote job listings from We Work Remotely",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'jobs.csv')
JSON_PATH = os.path.join(DATA_DIR, 'jobs.json')


def load_jobs():
    """Load jobs from CSV or JSON"""
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
    
    return []


@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "Job Scraper API",
        "version": "1.0.0",
        "endpoints": {
            "/jobs": "Get all jobs with optional filtering",
            "/jobs/search": "Search jobs by keyword",
            "/jobs/{job_id}": "Get specific job by ID",
            "/stats": "Get statistics about jobs"
        }
    }


@app.get("/jobs")
def get_jobs(
    company: Optional[str] = Query(None, description="Filter by company name"),
    location: Optional[str] = Query(None, description="Filter by location"),
    title_contains: Optional[str] = Query(None, description="Filter by job title keyword"),
    limit: int = Query(100, ge=1, le=1000, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip")
):
    """
    Get all jobs with optional filtering and pagination
    """
    jobs = load_jobs()
    
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found. Run the scraper first.")
    
    # Apply filters
    filtered_jobs = jobs
    
    if company:
        filtered_jobs = [j for j in filtered_jobs if company.lower() in str(j.get('company', '')).lower()]
    
    if location:
        filtered_jobs = [j for j in filtered_jobs if location.lower() in str(j.get('location', '')).lower()]
    
    if title_contains:
        filtered_jobs = [j for j in filtered_jobs if title_contains.lower() in str(j.get('job_title', '')).lower()]
    
    # Get total count before pagination
    total = len(filtered_jobs)
    
    # Apply pagination
    filtered_jobs = filtered_jobs[offset:offset + limit]
    
    return {
        "jobs": filtered_jobs,
        "total": total,
        "limit": limit,
        "offset": offset,
        "returned": len(filtered_jobs)
    }


@app.get("/jobs/search")
def search_jobs(
    q: str = Query(..., description="Search query"),
    limit: int = Query(50, ge=1, le=1000)
):
    """
    Search jobs by keyword in title, company, or location
    """
    jobs = load_jobs()
    
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    
    query = q.lower()
    results = []
    
    for job in jobs:
        text = f"{job.get('job_title', '')} {job.get('company', '')} {job.get('location', '')}".lower()
        if query in text:
            results.append(job)
    
    return {
        "query": q,
        "results": results[:limit],
        "total_found": len(results)
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: int):
    """
    Get a specific job by its index (ID)
    """
    jobs = load_jobs()
    
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    
    if job_id < 0 or job_id >= len(jobs):
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
    
    return jobs[job_id]


@app.get("/stats")
def get_stats():
    """
    Get statistics about the scraped jobs
    """
    jobs = load_jobs()
    
    if not jobs:
        return {
            "total_jobs": 0,
            "companies": 0,
            "locations": 0,
            "last_scraped": None
        }
    
    # Calculate stats
    companies = set(str(j.get('company', '')) for j in jobs if j.get('company'))
    locations = set(str(j.get('location', '')) for j in jobs if j.get('location'))
    
    # Get last scraped date
    dates = [j.get('date_scraped') for j in jobs if j.get('date_scraped')]
    last_scraped = max(dates) if dates else None
    
    return {
        "total_jobs": len(jobs),
        "companies": len(companies),
        "locations": len(locations),
        "last_scraped": last_scraped
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_file_exists": os.path.exists(CSV_PATH) or os.path.exists(JSON_PATH)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

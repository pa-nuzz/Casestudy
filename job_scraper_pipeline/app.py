from fastapi import FastAPI, HTTPException
import pandas as pd
import os

app = FastAPI()
CSV_PATH = os.path.join('data', 'jobs.csv')

@app.get('/jobs')
def get_jobs():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="jobs.csv not found. Please run the scraper first.")
    try:
        df = pd.read_csv(CSV_PATH)
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading jobs.csv: {e}")

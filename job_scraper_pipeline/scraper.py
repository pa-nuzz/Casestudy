"""
Simple Job Scraper for Fake Jobs Site
Scrapes job title, company, and location
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import random
import os
from datetime import datetime

# Target URL - Using fake-jobs site for demo (no Cloudflare)
URL = 'https://realpython.github.io/fake-jobs/'

# Headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
}

# Output paths
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'jobs.csv')
JSON_PATH = os.path.join(DATA_DIR, 'jobs.json')


def scrape_jobs():
    """
    Scrape jobs from the website
    Returns list of job dictionaries
    """
    print(f"Fetching jobs from: {URL}")
    
    try:
        # Make request
        response = requests.get(URL, headers=HEADERS, timeout=30)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: Failed to fetch page (Status {response.status_code})")
            return []
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find job cards - each job is in a div with class 'card'
        job_cards = soup.find_all('div', class_='card')
        
        print(f"Found {len(job_cards)} job cards")
        
        jobs = []
        for i, card in enumerate(job_cards):
            try:
                # Extract job title from h2 with class 'title is-5'
                title_elem = card.find('h2', class_='title is-5')
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                # Extract company from h3 with class 'subtitle is-6 company'
                company_elem = card.find('h3', class_='subtitle is-6 company')
                company = company_elem.get_text(strip=True) if company_elem else ''
                
                # Extract location from p with class 'location'
                location_elem = card.find('p', class_='location')
                location = location_elem.get_text(strip=True) if location_elem else 'Remote'
                
                # Get job URL from Apply link
                apply_link = card.find('a', string='Apply')
                job_url = apply_link['href'] if apply_link and apply_link.get('href') else ''
                
                # Only add if we have a valid title
                if title and company:
                    jobs.append({
                        'job_title': title,
                        'company': company,
                        'location': location,
                        'job_url': job_url,
                        'date_scraped': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'realpython.github.io/fake-jobs'
                    })
                    
            except Exception as e:
                print(f"Error parsing job {i}: {e}")
                continue
        
        print(f"Successfully parsed {len(jobs)} jobs")
        return jobs
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def save_jobs(jobs):
    """
    Save jobs to CSV and JSON
    """
    if not jobs:
        print("No jobs to save")
        return
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(jobs)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(jobs)} jobs to {CSV_PATH}")
    
    # Save to JSON
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(jobs)} jobs to {JSON_PATH}")
    
    # Show first 3 jobs as sample
    print("\nSample jobs:")
    for job in jobs[:3]:
        print(f"  - {job['job_title']} at {job['company']} ({job['location']})")


def run_pipeline():
    """
    Main pipeline function
    """
    print("="*60)
    print("Job Scraper Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Scrape jobs
    jobs = scrape_jobs()
    
    # Save jobs
    save_jobs(jobs)
    
    print()
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return jobs


if __name__ == '__main__':
    run_pipeline()

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import os

URL = 'https://weworkremotely.com/categories/remote-programming-jobs'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
OUTPUT_PATH = os.path.join('data', 'jobs.csv')


def scrape_jobs():
    response = requests.get(URL, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        print(response.text[:500])
        return

    print("First 500 chars of response:")
    print(response.text[:500])

    try:
        soup = BeautifulSoup(response.text, features='xml')
        items = soup.find_all('item')
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return

    jobs = []
    for item in items:
        try:
            title_tag = item.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ''
            desc_tag = item.find('description')
            description = desc_tag.get_text(strip=True) if desc_tag else ''
            company, location = '', ''
            if ' at ' in title:
                parts = title.split(' at ')
                job_title = parts[0]
                rest = parts[1]
                if ' (' in rest and rest.endswith(')'):
                    company = rest[:rest.rfind(' (')]
                    location = rest[rest.rfind(' (')+2:-1]
                else:
                    company = rest
            elif '-' in title:
                parts = title.split('-')
                job_title = parts[0].strip()
                company = parts[1].strip() if len(parts) > 1 else ''
            else:
                job_title = title
            if not location and ',' in description:
                location = description.split(',')[-1].strip()
            jobs.append({
                'job_title': job_title,
                'company': company,
                'location': location,
                'date_scraped': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            print(f"Error extracting job: {e}")
        time.sleep(0.1)

    jobs = [job for job in jobs if job['job_title']]
    df = pd.DataFrame(jobs)
    os.makedirs('data', exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} jobs to {OUTPUT_PATH}")

if __name__ == '__main__':
    try:
        scrape_jobs()
    except Exception as e:
        print(f"Error: {e}")

# Fallback: If the site uses JavaScript to load jobs, use Selenium:
# from selenium import webdriver
# ... (Selenium logic here)

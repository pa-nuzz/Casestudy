# Job Scraper Pipeline

This project provides a professional pipeline for scraping job listings and serving them via an API.

## Features
- Scrapes job title, company, and location from [Fake Jobs](https://realpython.github.io/fake-jobs/)
- Adds a `date_scraped` column
- Saves results to `data/jobs.csv`
- FastAPI endpoint to serve jobs as JSON
- Automated daily scraping and commit via GitHub Actions

## Anti-Scraping & Rate Limiting
- **User-Agent Rotation:** Always set a custom User-Agent header to avoid basic bot detection. For more advanced scraping, rotate User-Agents and consider using proxies.
- **Rate Limiting:** Use `time.sleep()` between requests to avoid overwhelming the server and reduce the risk of being blocked.
- **Proxies:** For sites with stricter anti-bot measures, use proxy services to rotate IP addresses.
- **Selenium Fallback:** If content is loaded dynamically (JavaScript), use Selenium to render pages before scraping.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the scraper: `python scraper.py`
3. Start the API: `uvicorn app:app --reload`

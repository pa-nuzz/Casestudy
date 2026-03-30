# Job Scraper Pipeline

A simple job scraper for We Work Remotely that scrapes job listings and serves them via FastAPI.

## Challenges & Anti-Scraping Notes

### Why We Use a Practice Site
This scraper uses `realpython.github.io/fake-jobs/` — a practice site designed for learning web scraping. This ensures the scraper works reliably for demos without dealing with anti-bot protections.

### Real-World Scraping Challenges

**1. Rate Limiting**
- **Challenge:** Sites block IPs making too many requests
- **Solution:** Add delays (1-3 seconds), rotate user agents, use exponential backoff

**2. Cloudflare Protection**
- **Challenge:** Cloudflare blocks automated requests with CAPTCHAs
- **Solutions:**
  - Use `cloudscraper` library to bypass
  - Use headless browsers (Selenium, Playwright) for JS-heavy sites
  - Respect robots.txt and terms of service

**3. Dynamic Content (JavaScript Rendering)**
- **Challenge:** Jobs load via AJAX/JavaScript
- **Solutions:**
  - Use Selenium or Playwright for full browser automation
  - Check for hidden API endpoints (Network tab in DevTools)
  - Look for RSS feeds or sitemap.xml

**4. Login Walls & Authentication**
- **Challenge:** Sites require login to view jobs
- **Solutions:**
  - Use session cookies with requests
  - OAuth token handling
  - Ethical consideration: only scrape public data

**5. Ethical Considerations**
- Always check `robots.txt` before scraping
- Respect rate limits (don't overwhelm servers)
- Consider using official APIs when available
- Scrape responsibly and legally
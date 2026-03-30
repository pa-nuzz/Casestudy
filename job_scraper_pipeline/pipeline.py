"""
Daily Job Scraper Pipeline
Runs the scraper daily using APScheduler
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from scraper import run_pipeline
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def daily_job():
    """
    Job that runs daily to scrape new jobs
    """
    logger.info("Starting daily job scraping...")
    try:
        jobs = run_pipeline()
        logger.info(f"Successfully scraped {len(jobs)} jobs")
    except Exception as e:
        logger.error(f"Error in daily job: {e}")


def start_scheduler():
    """
    Start the scheduler to run daily at 9 AM
    """
    scheduler = BlockingScheduler()
    
    # Run daily at 9:00 AM
    scheduler.add_job(
        daily_job,
        trigger=CronTrigger(hour=9, minute=0),
        id='daily_scraper',
        name='Daily Job Scraper',
        replace_existing=True
    )
    
    logger.info("Scheduler started. Running daily at 9:00 AM")
    logger.info("Press Ctrl+C to exit")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == '__main__':
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    
    # Run once immediately, then schedule
    logger.info("Running initial scrape...")
    daily_job()
    
    # Start daily scheduler
    start_scheduler()

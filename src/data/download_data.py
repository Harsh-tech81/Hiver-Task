"""
Downloads the raw dataset from Kaggle.
"""
import os
import logging
from pathlib import Path
import sys

try:
    from config import RAW_DATA_DIR
except ImportError:
    RAW_DATA_DIR = Path("data/raw")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def download_dataset() -> None:
    """
    Downloads the thoughtvector/customer-support-on-twitter dataset using opendatasets.
    Falls back to instructions if credentials are missing.
    """
    try:
        import opendatasets as od
    except ImportError:
        logger.error("opendatasets not installed. Run: pip install opendatasets")
        return

    dataset_url = "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter"
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dataset to {raw_dir}...")
    try:
        od.download(dataset_url, data_dir=str(raw_dir))
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Please download manually from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        logger.info(f"Extract twcs.csv and place it in {raw_dir}")

if __name__ == "__main__":
    download_dataset()

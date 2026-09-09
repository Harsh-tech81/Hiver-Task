import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_DATA_DIR = DATA_DIR / "golden"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, GOLDEN_DATA_DIR, RESULTS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# Model Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEMPERATURE = 0.3
MAX_TOKENS = 512

# Data Configuration
TARGET_BRAND = "AppleSupport"
MAX_CONVERSATIONS = 10000  # Subsample size
MIN_THREAD_LENGTH = 2  # Min messages in a thread
MAX_THREAD_LENGTH = 20  # Max messages in a thread

# Retrieval Configuration
TOP_K_RETRIEVAL = 5
FAISS_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.npy"

# Evaluation Configuration
GOLDEN_SET_SIZE = 200
JUDGE_SAMPLE_SIZE = 50

# Intent Categories
INTENT_CATEGORIES = [
    "device_troubleshoot",
    "account_access",
    "billing_payment",
    "warranty_repair",
    "app_issue",
    "connectivity",
    "data_recovery",
    "general_inquiry",
    "feedback_complaint",
    "other",
]

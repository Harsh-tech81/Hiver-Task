#!/usr/bin/env python3
"""Main pipeline script for the AI Customer Support Agent.

Run this to reproduce all results end-to-end.
Usage:
    python run_pipeline.py              # Full pipeline
    python run_pipeline.py --skip-download  # Skip data download (if already downloaded)
    python run_pipeline.py --eval-only     # Only run evaluation (data must exist)
    python run_pipeline.py --demo          # Run interactive demo
"""

import argparse
import json
import logging
import time
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, GOLDEN_DATA_DIR,
    RESULTS_DIR, MODEL_DIR, FAISS_INDEX_PATH, EMBEDDINGS_PATH
)
from src.data.download_data import download_dataset
from src.data.preprocess import preprocess_pipeline
from src.intent.classifier import KeywordClassifier, TfidfClassifier, LLMClassifier
from src.reply.retriever import ConversationRetriever
from src.agent import SupportAgent
from src.eval.build_golden_set import build_golden_set
from src.eval.evaluate import run_full_evaluation
from src.eval.judge_agreement import run_agreement_analysis

CONVERSATIONS_PATH = PROCESSED_DATA_DIR / "apple_conversations.jsonl"
GOLDEN_SET_PATH = GOLDEN_DATA_DIR / "golden_eval_set.jsonl"
CONVERSATIONS_METADATA_PATH = MODEL_DIR / "conversations_metadata.pkl"


def timer(step_name: str):
    """Decorator to log start/end times and handle errors for pipeline steps."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"--- Starting Step: {step_name} ---")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"--- Finished Step: {step_name} in {elapsed:.2f} seconds ---")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"--- Error in Step: {step_name} after {elapsed:.2f} seconds. Error: {e} ---")
                import traceback
                traceback.print_exc()
                return None
        return wrapper
    return decorator


@timer("Data Download")
def step_download_data() -> None:
    """Download dataset from Kaggle."""
    download_dataset()


@timer("Data Preprocessing")
def step_preprocess_data(max_conversations: int) -> None:
    """Preprocess raw data into conversation threads."""
    if CONVERSATIONS_PATH.exists():
        logger.info(f"Processed data already exists at {CONVERSATIONS_PATH}. Skipping.")
        return
    preprocess_pipeline(max_conversations)


@timer("Build Retrieval Index")
def step_build_index() -> None:
    """Build FAISS index for conversation retrieval."""
    if Path(FAISS_INDEX_PATH).exists():
        logger.info(f"FAISS index already exists at {FAISS_INDEX_PATH}. Skipping build.")
        return

    if not CONVERSATIONS_PATH.exists():
        logger.error(f"Processed data not found at {CONVERSATIONS_PATH}. Cannot build index.")
        return

    # Load conversations
    conversations = []
    with open(CONVERSATIONS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    logger.info(f"Building FAISS index from {len(conversations)} conversations...")
    retriever = ConversationRetriever()
    retriever.build_index(conversations)
    retriever.save_index(
        str(FAISS_INDEX_PATH),
        str(EMBEDDINGS_PATH),
        str(CONVERSATIONS_METADATA_PATH)
    )
    logger.info("FAISS index built and saved successfully.")


@timer("Train TF-IDF Classifier")
def step_train_tfidf() -> None:
    """Train TF-IDF classifier using golden set labels."""
    tfidf_model_path = MODEL_DIR / "tfidf_classifier.joblib"
    if tfidf_model_path.exists():
        logger.info(f"TF-IDF model already exists at {tfidf_model_path}. Skipping.")
        return

    if not GOLDEN_SET_PATH.exists():
        logger.warning("Golden set not found. Cannot train TF-IDF. Skipping.")
        return

    # Load golden set for training data
    golden_data = []
    with open(GOLDEN_SET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                golden_data.append(json.loads(line))

    texts = [item['customer_message'] for item in golden_data if item.get('customer_message')]
    labels = [item['gold_intent'] for item in golden_data if item.get('customer_message')]

    if len(texts) < 10:
        logger.warning("Not enough training data for TF-IDF classifier. Skipping.")
        return

    classifier = TfidfClassifier()
    classifier.train(texts, labels)
    classifier.save(str(tfidf_model_path))
    logger.info(f"TF-IDF classifier trained and saved to {tfidf_model_path}")


@timer("Build Golden Set")
def step_build_golden_set() -> None:
    """Build golden evaluation set with auto-labeling."""
    if GOLDEN_SET_PATH.exists():
        logger.info(f"Golden set already exists at {GOLDEN_SET_PATH}. Skipping.")
        return
    build_golden_set()


@timer("Run Evaluation")
def step_evaluate() -> dict:
    """Run full evaluation with baseline comparison."""
    if not GOLDEN_SET_PATH.exists():
        logger.error("Golden set not found. Cannot evaluate. Skipping.")
        return {}
    return run_full_evaluation(GOLDEN_SET_PATH)


@timer("Judge Agreement Analysis")
def step_judge_agreement() -> dict:
    """Run judge agreement analysis."""
    if not GOLDEN_SET_PATH.exists():
        logger.error("Golden set not found. Cannot run judge agreement. Skipping.")
        return {}
    return run_agreement_analysis(GOLDEN_SET_PATH)


def run_demo() -> None:
    """Run interactive demo mode."""
    print("\n" + "=" * 60)
    print("  AI Customer Support Agent - Interactive Demo")
    print("  Brand: Apple Support (@AppleSupport)")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    # Initialize agent with the main model configuration
    try:
        agent = SupportAgent(
            classifier_type='llm',
            generator_type='rag',
            escalation_type='hybrid'
        )
        # Try to load retriever index
        try:
            agent.retriever.load_index(
                str(FAISS_INDEX_PATH),
                str(EMBEDDINGS_PATH),
                str(CONVERSATIONS_METADATA_PATH)
            )
            logger.info("Loaded FAISS retriever index.")
        except Exception as e:
            logger.warning(f"Could not load retriever index: {e}")
            logger.warning("Replies will not be grounded in historical data.")
    except Exception as e:
        logger.error(f"Failed to initialize main agent: {e}")
        logger.info("Falling back to keyword + template agent...")
        agent = SupportAgent(
            classifier_type='keyword',
            generator_type='template',
            escalation_type='rules'
        )

    history = []
    while True:
        try:
            msg = input("\n🧑 Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not msg or msg.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        try:
            response = agent.process_message(msg, history)

            print(f"\n📋 Intent: {response.intent} (confidence: {response.intent_confidence:.2f})")
            print(f"🤖 Reply: {response.reply}")
            esc_icon = "🚨" if response.escalation_decision == 'escalate' else "✅"
            print(f"{esc_icon} Escalation: {response.escalation_decision.upper()} — {response.escalation_reason}")

            history.append({"role": "user", "content": msg, "is_customer": True, "text": msg})
            history.append({"role": "agent", "content": response.reply, "is_customer": False, "text": response.reply})
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Customer Support Agent — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                     # Full pipeline
  python run_pipeline.py --skip-download     # Skip Kaggle download
  python run_pipeline.py --eval-only         # Only run evaluation
  python run_pipeline.py --demo              # Interactive demo
        """
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip the data download step")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (assumes data exists)")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo mode")
    parser.add_argument("--max-conversations", type=int, default=5000,
                        help="Max conversations to process (default: 5000)")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  AI Customer Support Agent — Pipeline")
    logger.info("=" * 60)

    if not args.eval_only:
        # Step 1: Download data
        if not args.skip_download:
            step_download_data()

        # Step 2: Preprocess data
        step_preprocess_data(args.max_conversations)

        # Step 3: Build FAISS index
        step_build_index()

        # Step 4: Build golden evaluation set
        step_build_golden_set()

        # Step 5: Train TF-IDF classifier
        step_train_tfidf()

    # Step 6: Run evaluation
    step_evaluate()

    # Step 7: Judge agreement analysis
    step_judge_agreement()

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"  Pipeline completed in {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"  Results saved to: {RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

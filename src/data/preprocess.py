"""
Preprocesses the raw customer support dataset.
"""
import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

try:
    from config import (
        RAW_DATA_DIR, 
        PROCESSED_DATA_DIR, 
        TARGET_BRAND, 
        MAX_CONVERSATIONS, 
        MIN_THREAD_LENGTH, 
        MAX_THREAD_LENGTH
    )
except ImportError:
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    TARGET_BRAND = 'AppleSupport'
    MAX_CONVERSATIONS = 10000
    MIN_THREAD_LENGTH = 2
    MAX_THREAD_LENGTH = 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Loads the raw CSV dataset."""
    logger.info(f"Loading raw data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        raise

def identify_brand_conversations(df: pd.DataFrame, brand: str = 'AppleSupport') -> pd.DataFrame:
    """Filters data to conversations involving the specified brand."""
    logger.info(f"Filtering conversations for brand: {brand}...")
    brand_tweets = df[df['author_id'] == brand]
    
    brand_authored = set(brand_tweets['tweet_id'].dropna().astype(str))
    brand_responded_to = set(brand_tweets['in_response_to_tweet_id'].dropna().astype(int).astype(str))
    
    df['tweet_id_str'] = df['tweet_id'].astype(str)
    df['in_response_to_tweet_id_str'] = df['in_response_to_tweet_id'].apply(
        lambda x: str(int(x)) if pd.notnull(x) else None
    )
    
    mask_author = df['author_id'] == brand
    mask_reply_to_brand = df['in_response_to_tweet_id_str'].isin(brand_authored)
    mask_brand_responded_to = df['tweet_id_str'].isin(brand_responded_to)
    
    filtered_df = df[mask_author | mask_reply_to_brand | mask_brand_responded_to].copy()
    logger.info(f"Filtered to {len(filtered_df)} tweets related to {brand}.")
    return filtered_df

def clean_text(text: str) -> str:
    """
    Cleans tweet text: removes @mentions, URLs, normalizes whitespace.
    Handles __EMAIL__ and __PHONE__ tokens.
    """
    if pd.isnull(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_conversation_threads(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Reconstructs threads by tracing in_response_to_tweet_id chains.
    Optimized for high performance and low memory.
    """
    from collections import deque
    logger.info("Building conversation threads (optimized)...")
    
    df = df.copy()
    try:
        df['parsed_dt'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
        df = df.sort_values('parsed_dt')
    except Exception:
        pass

    tweet_ids = set(df['tweet_id_str'])
    tweet_authors = dict(zip(df['tweet_id_str'], df['author_id'].astype(str)))
    tweet_inbound = dict(zip(df['tweet_id_str'], df['inbound'].astype(bool)))
    tweet_texts = dict(zip(df['tweet_id_str'], df['text'].astype(str)))
    tweet_times = dict(zip(df['tweet_id_str'], df['created_at'].astype(str)))
    
    child_to_parent = df[['tweet_id_str', 'in_response_to_tweet_id_str']].dropna()
    child_to_parent_dict = dict(zip(child_to_parent['tweet_id_str'], child_to_parent['in_response_to_tweet_id_str']))
    
    parent_to_children = {}
    for child, parent in child_to_parent_dict.items():
        if parent in tweet_ids:
            parent_to_children.setdefault(parent, []).append(child)
        
    roots = [tid for tid in tweet_ids if child_to_parent_dict.get(tid) not in tweet_ids]
            
    threads = []
    visited = set()
    
    for root_id in roots:
        if root_id in visited:
            continue
            
        thread_messages = []
        queue = deque([root_id])
        
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            
            thread_messages.append({
                'tweet_id': curr,
                'author_id': tweet_authors.get(curr, ''),
                'is_customer': tweet_inbound.get(curr, True),
                'text': tweet_texts.get(curr, ''),
                'timestamp': tweet_times.get(curr, '')
            })
            
            children = parent_to_children.get(curr)
            if children:
                queue.extend(children)
            
        customer_msgs = [m['text'] for m in thread_messages if m['is_customer']]
        agent_msgs = [m['text'] for m in thread_messages if not m['is_customer']]
        
        thread = {
            'thread_id': root_id,
            'messages': thread_messages,
            'customer_messages': customer_msgs,
            'agent_responses': agent_msgs,
            'num_turns': len(thread_messages)
        }
        threads.append(thread)
        
    logger.info(f"Built {len(threads)} threads.")
    return threads

def preprocess_pipeline(max_conversations: int = MAX_CONVERSATIONS) -> None:
    """Orchestrates the full preprocessing pipeline."""
    logger.info("Starting preprocessing pipeline...")
    
    candidates = [
        Path(RAW_DATA_DIR) / 'thoughtvector' / 'customer-support-on-twitter' / 'twcs.csv',
        Path(RAW_DATA_DIR) / 'twcs.csv',
        Path("src/data/raw/twcs.csv"),
        Path(__file__).parent / 'raw' / 'twcs.csv',
    ]
    raw_path = None
    for p in candidates:
        if p.exists():
            raw_path = p
            break
        
    if not raw_path:
        logger.error(f"Raw data file not found in candidates: {[str(c) for c in candidates]}. Run download_data.py first.")
        return
        
    df = load_raw_data(raw_path)
    
    df_filtered = identify_brand_conversations(df, brand=TARGET_BRAND)
    
    threads = build_conversation_threads(df_filtered)
    
    for t in threads:
        for m in t['messages']:
            m['text'] = clean_text(m['text'])
        t['customer_messages'] = [clean_text(msg) for msg in t['customer_messages']]
        t['agent_responses'] = [clean_text(msg) for msg in t['agent_responses']]
        
    valid_threads = [
        t for t in threads 
        if MIN_THREAD_LENGTH <= t['num_turns'] <= MAX_THREAD_LENGTH
    ]
    logger.info(f"Filtered to {len(valid_threads)} threads based on length ({MIN_THREAD_LENGTH}-{MAX_THREAD_LENGTH}).")
    
    if len(valid_threads) > max_conversations:
        import random
        random.seed(42)
        valid_threads = random.sample(valid_threads, max_conversations)
        logger.info(f"Subsampled to {max_conversations} threads.")
        
    out_dir = Path(PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'apple_conversations.jsonl'
    
    logger.info(f"Saving processed data to {out_file}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        for t in valid_threads:
            f.write(json.dumps(t) + '\n')
            
    logger.info("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_pipeline()

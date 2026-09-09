import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

import openai

from config import PROCESSED_DATA_DIR, GOLDEN_DATA_DIR, GOLDEN_SET_SIZE, INTENT_CATEGORIES, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from src.intent.classifier import KeywordClassifier, LLMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def sample_conversations(conversations: List[Dict], n: int = 200) -> List[Dict]:
    """
    Samples conversations using stratified sampling to ensure at least 15 examples per intent category.
    
    Args:
        conversations: List of all loaded conversations.
        n: Total number of samples required.
        
    Returns:
        List[Dict]: Sampled conversations.
    """
    random.seed(42)
    classifier = KeywordClassifier()
    
    intent_groups: Dict[str, List[Dict]] = {category: [] for category in INTENT_CATEGORIES}
    unclassified = []
    
    for conv in conversations:
        first_msg = conv.get('customer_messages', [''])[0] if conv.get('customer_messages') else ''
        if not first_msg:
            continue
            
        intent, _ = classifier.classify(first_msg)
        if intent in intent_groups:
            intent_groups[intent].append(conv)
        else:
            unclassified.append(conv)
            
    sampled = []
    for intent, group in intent_groups.items():
        k = min(15, len(group))
        sampled.extend(random.sample(group, k))
        for s in sampled[-k:]:
            group.remove(s)
            
    remaining_needed = n - len(sampled)
    if remaining_needed > 0:
        pool = []
        for group in intent_groups.values():
            pool.extend(group)
        pool.extend(unclassified)
        
        if len(pool) >= remaining_needed:
            sampled.extend(random.sample(pool, remaining_needed))
        else:
            sampled.extend(pool)
            
    random.shuffle(sampled)
    return sampled

def auto_label_with_llm(conversations: List[Dict], max_workers: int = 6) -> List[Dict]:
    """
    Uses OpenAI to auto-generate golden labels for each conversation.
    Uses a unified prompt (intent + escalation + quality) and ThreadPoolExecutor for speed.
    
    Args:
        conversations: Sampled conversations.
        max_workers: Number of concurrent worker threads.
        
    Returns:
        List[Dict]: Labeled conversations.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    keyword_fallback = KeywordClassifier()
    client_kwargs = {'api_key': OPENAI_API_KEY} if OPENAI_API_KEY else {}
    if OPENAI_BASE_URL:
        client_kwargs['base_url'] = OPENAI_BASE_URL
    client = openai.OpenAI(**client_kwargs) if OPENAI_API_KEY else None
    
    intents_str = ", ".join(INTENT_CATEGORIES)
    labeled_data = [None] * len(conversations)

    def label_single(idx: int, conv: Dict) -> Dict:
        first_msg = conv.get('customer_messages', [''])[0] if conv.get('customer_messages') else ''
        apple_response = conv.get('agent_responses', [''])[0] if conv.get('agent_responses') else 'No response'
        
        gold_intent, _ = keyword_fallback.classify(first_msg)
        gold_escalation = False
        gold_escalation_reason = "Can be resolved using standard procedures"
        gold_reply_quality = 4
        
        if client and first_msg:
            prompt = f"""You are an expert customer support evaluator for Apple Support.
Analyze the following customer message and actual agent response.

Allowed Intent Categories: {intents_str}

Customer Message: "{first_msg}"
Agent Response: "{apple_response}"

Respond strictly with a JSON object:
{{
    "gold_intent": "<one of the allowed intent categories>",
    "gold_escalation": <true or false>,
    "gold_escalation_reason": "<short explanation>",
    "gold_reply_quality": <integer 1-5 rating the agent response>
}}"""
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.0
                    )
                    content = resp.choices[0].message.content
                    parsed = json.loads(content)
                    pred_intent = parsed.get("gold_intent", "")
                    if pred_intent in INTENT_CATEGORIES:
                        gold_intent = pred_intent
                    gold_escalation = bool(parsed.get("gold_escalation", False))
                    gold_escalation_reason = str(parsed.get("gold_escalation_reason", gold_escalation_reason))
                    gold_reply_quality = int(parsed.get("gold_reply_quality", 4))
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(f"Failed LLM labeling for conv {conv.get('thread_id')}: {e}. Used fallback.")
        
        return {
            "id": conv.get("thread_id", str(idx)),
            "customer_message": first_msg,
            "conversation_context": conv.get("messages", []),
            "gold_intent": gold_intent,
            "gold_reply_reference": apple_response,
            "gold_escalation": gold_escalation,
            "gold_escalation_reason": gold_escalation_reason,
            "gold_reply_quality": gold_reply_quality
        }

    logger.info(f"Labeling {len(conversations)} conversations using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(label_single, i, conv): i for i, conv in enumerate(conversations)}
        for future in tqdm(as_completed(futures), total=len(conversations), desc="Auto-labeling Golden Set"):
            i = futures[future]
            try:
                labeled_data[i] = future.result()
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                
    return [item for item in labeled_data if item is not None]

def build_golden_set(conversations_path: Path = None, output_path: Path = None):
    """
    Builds the golden evaluation set from raw conversations and saves it.
    
    Args:
        conversations_path: Path to raw conversations JSONL.
        output_path: Path to save golden set.
    """
    if conversations_path is None:
        conversations_path = Path(PROCESSED_DATA_DIR) / 'apple_conversations.jsonl'
        
    if output_path is None:
        output_path = Path(GOLDEN_DATA_DIR) / 'golden_eval_set.jsonl'
        
    logger.info(f"Loading conversations from {conversations_path}")
    conversations = []
    if conversations_path.exists():
        with open(conversations_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))
    else:
        logger.error(f"File not found: {conversations_path}")
        return
        
    logger.info(f"Loaded {len(conversations)} conversations. Sampling {GOLDEN_SET_SIZE}...")
    sampled = sample_conversations(conversations, n=GOLDEN_SET_SIZE)
    
    logger.info("Auto-labeling with LLM...")
    labeled = auto_label_with_llm(sampled)
    
    logger.info(f"Saving golden set to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in labeled:
            f.write(json.dumps(item) + '\n')
            
    stats_path = output_path.parent / 'golden_eval_stats.json'
    intent_counts = {}
    escalation_count = 0
    for item in labeled:
        intent = item['gold_intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        if item['gold_escalation']:
            escalation_count += 1
            
    stats = {
        'total_examples': len(labeled),
        'intent_distribution': intent_counts,
        'escalation_ratio': escalation_count / len(labeled) if len(labeled) > 0 else 0
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
        
    logger.info(f"Summary stats saved to {stats_path}")

if __name__ == '__main__':
    build_golden_set()

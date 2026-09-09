import json
import logging
from typing import Dict, List, Any
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import random
import os
import sys

# Try to import config, assuming it's in the project root
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import config
except ImportError:
    config = None

# Import LLMJudge from local module
from .llm_judge import LLMJudge

logger = logging.getLogger(__name__)

def compute_judge_agreement(human_scores: List[int], llm_scores: List[int]) -> Dict[str, float]:
    """
    Compute agreement metrics between human and LLM judge scores.
    """
    if not human_scores or not llm_scores or len(human_scores) != len(llm_scores):
        return {}
        
    kappa = cohen_kappa_score(human_scores, llm_scores)
    spearman_corr, _ = spearmanr(human_scores, llm_scores)
    pearson_corr, _ = pearsonr(human_scores, llm_scores)
    mae = mean_absolute_error(human_scores, llm_scores)
    
    exact_match = sum(1 for h, l in zip(human_scores, llm_scores) if h == l) / len(human_scores)
    within_one = sum(1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= 1) / len(human_scores)
    
    return {
        'cohen_kappa': float(kappa),
        'spearman_correlation': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'pearson_correlation': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'mean_absolute_error': float(mae),
        'agreement_exact': float(exact_match),
        'agreement_within_1_point': float(within_one)
    }

def simulate_human_labels(golden_set: List[Dict[str, Any]], sample_size: int = 50) -> List[Dict[str, Any]]:
    """
    Sample golden set to act as human labels.
    """
    if config and hasattr(config, 'JUDGE_SAMPLE_SIZE'):
        sample_size = config.JUDGE_SAMPLE_SIZE
        
    sampled = random.sample(golden_set, min(sample_size, len(golden_set)))
    
    human_labels = []
    for item in sampled:
        human_labels.append({
            'id': item.get('id', ''),
            'customer_message': item.get('customer_message', ''),
            'reference_reply': item.get('gold_reply_reference', ''),
            'generated_reply': item.get('gold_reply_reference', ''), # For self-evaluation
            'intent': item.get('gold_intent', ''),
            'human_score': item.get('gold_reply_quality', 3)
        })
    return human_labels

def run_agreement_analysis(golden_set_path: Path = None) -> Dict[str, Any]:
    """
    Run agreement analysis between human scores (from golden set) and LLM judge.
    """
    if golden_set_path is None:
        if config and hasattr(config, 'GOLDEN_DATA_DIR'):
            golden_set_path = Path(config.GOLDEN_DATA_DIR) / 'golden_eval_set.jsonl'
        else:
            logger.error("golden_set_path not provided and config not available.")
            return {}
            
    if not golden_set_path.exists():
        logger.error(f"Golden set not found at {golden_set_path}")
        return {}
        
    golden_set = []
    with open(golden_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                golden_set.append(json.loads(line))
                
    simulated = simulate_human_labels(golden_set)
    
    judge = LLMJudge()
    human_scores = []
    llm_scores = []
    
    for item in simulated:
        human_scores.append(item['human_score'])
        
    llm_results = judge.judge_batch(simulated)
    
    for res in llm_results:
        # Round overall score to nearest int for agreement
        llm_scores.append(round(res.get('overall', 3)))
        
    agreement_metrics = compute_judge_agreement(human_scores, llm_scores)
    
    results = {
        'agreement_metrics': agreement_metrics,
        'sample_size': len(simulated)
    }
    
    if config and hasattr(config, 'RESULTS_DIR'):
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / 'judge_agreement.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_agreement_analysis()

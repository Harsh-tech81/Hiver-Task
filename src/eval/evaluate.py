import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import os
import sys

# Try to import project modules
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import config
    from src.agent import SupportAgent
    from src.eval.metrics import compute_intent_metrics, compute_text_similarity_metrics, compute_escalation_metrics, format_results_table
    from src.eval.llm_judge import LLMJudge
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

def load_golden_set(path: Path = None) -> List[Dict[str, Any]]:
    """Load the golden evaluation dataset."""
    if path is None:
        path = Path(config.GOLDEN_DATA_DIR) / 'golden_eval_set.jsonl'
        
    data = []
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        logger.error(f"Golden set file not found at {path}")
        
    return data

def evaluate_model(agent, golden_set: List[Dict], model_name: str = 'main') -> Dict:
    """Evaluate an agent against the golden set."""
    logger.info(f"Evaluating model: {model_name} on {len(golden_set)} examples")
    
    predictions = {
        'intents': [],
        'replies': [],
        'escalations': []
    }
    
    gold = {
        'intents': [],
        'replies': [],
        'escalations': []
    }
    
    judge_examples = []
    
    for item in golden_set:
        customer_msg = item.get('customer_message', '')
        history = item.get('conversation_context', [])
        
        # Run agent
        try:
            response = agent.process_message(customer_msg, history)
            
            pred_intent = getattr(response, 'intent', 'unknown')
            pred_reply = getattr(response, 'reply', '')
            pred_escalation = getattr(response, 'escalation_decision', False)
        except Exception as e:
            logger.error(f"Agent failed on item {item.get('id')}: {e}")
            pred_intent, pred_reply, pred_escalation = 'unknown', '', False
            
        predictions['intents'].append(pred_intent)
        predictions['replies'].append(pred_reply)
        predictions['escalations'].append(str(pred_escalation))
        
        gold['intents'].append(item.get('gold_intent', ''))
        gold['replies'].append(item.get('gold_reply_reference', ''))
        gold['escalations'].append(item.get('gold_escalation', False))
        
        # Collect for LLM Judge (sample)
        judge_examples.append({
            'customer_message': customer_msg,
            'generated_reply': pred_reply,
            'reference_reply': item.get('gold_reply_reference', ''),
            'intent': item.get('gold_intent', '')
        })

    # Compute Metrics
    intent_metrics = compute_intent_metrics(predictions['intents'], gold['intents'])
    similarity_metrics = compute_text_similarity_metrics(predictions['replies'], gold['replies'])
    escalation_metrics = compute_escalation_metrics(predictions['escalations'], gold['escalations'])
    
    # LLM Judge on sample
    sample_size = getattr(config, 'JUDGE_SAMPLE_SIZE', 50)
    import random
    sampled_judge_examples = random.sample(judge_examples, min(sample_size, len(judge_examples)))
    
    judge = LLMJudge()
    judge_results = judge.judge_batch(sampled_judge_examples)
    
    avg_judge_scores = {
        'helpfulness': sum(r.get('helpfulness', 0) for r in judge_results) / len(judge_results) if judge_results else 0,
        'tone': sum(r.get('tone', 0) for r in judge_results) / len(judge_results) if judge_results else 0,
        'accuracy': sum(r.get('accuracy', 0) for r in judge_results) / len(judge_results) if judge_results else 0,
        'groundedness': sum(r.get('groundedness', 0) for r in judge_results) / len(judge_results) if judge_results else 0,
        'overall': sum(r.get('overall', 0) for r in judge_results) / len(judge_results) if judge_results else 0
    }
    
    return {
        'intent_metrics': intent_metrics,
        'similarity_metrics': similarity_metrics,
        'escalation_metrics': escalation_metrics,
        'llm_judge_scores': avg_judge_scores
    }

def run_baseline_comparison(golden_set: List[Dict]) -> Dict:
    """Compare multiple model configurations."""
    results = {}
    
    # Baseline 1: Trivial (keyword + nearest neighbor + rules)
    try:
        logger.info("Evaluating TRIVIAL baseline (keyword + nearest + rules)...")
        trivial_agent = SupportAgent(
            classifier_type='keyword',
            generator_type='nearest',
            escalation_type='rules'
        )
        try:
            trivial_agent.load_retriever()
        except Exception:
            logger.warning("Could not load retriever for trivial baseline.")
        results['trivial'] = evaluate_model(trivial_agent, golden_set, 'trivial')
    except Exception as e:
        logger.error(f"Trivial baseline failed: {e}")
    
    # Baseline 2: Simple (keyword + template + rules)
    try:
        logger.info("Evaluating SIMPLE baseline (keyword + template + rules)...")
        simple_agent = SupportAgent(
            classifier_type='keyword',
            generator_type='template',
            escalation_type='rules'
        )
        results['simple'] = evaluate_model(simple_agent, golden_set, 'simple')
    except Exception as e:
        logger.error(f"Simple baseline failed: {e}")
    
    # Main model (LLM + RAG + hybrid)
    try:
        logger.info("Evaluating MAIN model (LLM + RAG + hybrid)...")
        main_agent = SupportAgent(
            classifier_type='llm',
            generator_type='rag',
            escalation_type='hybrid'
        )
        try:
            main_agent.load_retriever()
        except Exception:
            logger.warning("Could not load retriever for main model.")
        results['main'] = evaluate_model(main_agent, golden_set, 'main')
    except Exception as e:
        logger.error(f"Main model failed: {e}")
        
    return results

def run_full_evaluation(golden_set_path: Path = None):
    """Run full evaluation and save results."""
    golden_set = load_golden_set(golden_set_path)
    if not golden_set:
        logger.error("Empty or missing golden set. Aborting evaluation.")
        return {}
        
    comparison = run_baseline_comparison(golden_set)
    
    results_dir = Path(getattr(config, 'RESULTS_DIR', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2)
        
    for model_name, res in comparison.items():
        print(f"\n--- {model_name.upper()} MODEL ---")
        print(format_results_table(res))
        
    return comparison

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Evaluate Hiver Support Agent")
    parser.add_argument('--golden-set', type=str, help="Path to golden set JSONL")
    parser.add_argument('--model', type=str, default='main', help="Model to evaluate")
    args = parser.parse_args()
    
    golden_path = Path(args.golden_set) if args.golden_set else None
    run_full_evaluation(golden_path)

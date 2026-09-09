import json
import logging
from typing import Dict, List, Any
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate as hf_evaluate

logger = logging.getLogger(__name__)

def compute_intent_metrics(predictions: List[str], gold_labels: List[str]) -> Dict[str, Any]:
    """
    Compute classification metrics for intent prediction.
    """
    labels = list(set(gold_labels) | set(predictions))
    
    accuracy = accuracy_score(gold_labels, predictions)
    precision_macro = precision_score(gold_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(gold_labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(gold_labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(gold_labels, predictions, average='weighted', zero_division=0)
    
    per_class_f1_arr = f1_score(gold_labels, predictions, labels=labels, average=None, zero_division=0)
    per_class_f1 = {label: float(f1) for label, f1 in zip(labels, per_class_f1_arr)}
    
    cm = confusion_matrix(gold_labels, predictions, labels=labels)
    
    return {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'per_class_f1': per_class_f1,
        'confusion_matrix': cm.tolist(),
        'labels': labels
    }

def compute_text_similarity_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    """
    Compute similarity metrics (BLEU, ROUGE-L, BERTScore) for text generation.
    """
    results = {}
    
    # BLEU
    try:
        smoothie = SmoothingFunction().method4
        refs_tokens = [[ref.split()] for ref in references]
        preds_tokens = [pred.split() for pred in predictions]
        bleu = corpus_bleu(refs_tokens, preds_tokens, smoothing_function=smoothie)
        results['bleu'] = float(bleu)
    except Exception as e:
        logger.error(f"Error computing BLEU: {e}")
        results['bleu'] = 0.0
        
    # ROUGE
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
        results['rouge_l_f1'] = float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0
    except Exception as e:
        logger.error(f"Error computing ROUGE: {e}")
        results['rouge_l_f1'] = 0.0
        
    # BERTScore
    try:
        bertscore = hf_evaluate.load("bertscore")
        bert_results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
        results['bertscore_f1'] = float(sum(bert_results['f1']) / len(bert_results['f1'])) if bert_results['f1'] else 0.0
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        results['bertscore_f1'] = 0.0
        
    return results

def compute_escalation_metrics(predictions: List[str], gold_labels: List[bool]) -> Dict[str, Any]:
    """
    Compute binary classification metrics for escalation decision.
    """
    pred_bools = [str(p).lower() == 'escalate' or str(p).lower() == 'true' for p in predictions]
    
    accuracy = accuracy_score(gold_labels, pred_bools)
    precision = precision_score(gold_labels, pred_bools, zero_division=0)
    recall = recall_score(gold_labels, pred_bools, zero_division=0)
    f1 = f1_score(gold_labels, pred_bools, zero_division=0)
    cm = confusion_matrix(gold_labels, pred_bools)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

def format_results_table(results: Dict) -> str:
    """
    Format results dictionary as a readable table.
    """
    output = "Evaluation Results\n"
    output += "="*50 + "\n"
    for category, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        output += f"\n{category.upper()}\n"
        output += "-"*30 + "\n"
        for metric, value in metrics.items():
            if isinstance(value, float):
                output += f"{metric:<20}: {value:.4f}\n"
            elif not isinstance(value, (list, dict)):
                output += f"{metric:<20}: {value}\n"
    return output

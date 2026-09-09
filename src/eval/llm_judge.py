import json
import logging
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from openai import OpenAI
import os
import sys

# Try to import config, assuming it's in the project root
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import config
except ImportError:
    config = None

logger = logging.getLogger(__name__)

class LLMJudge:
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or (config.LLM_MODEL if config else 'openai/gpt-4o-mini')
        self.api_key = api_key or (config.OPENAI_API_KEY if config else os.environ.get("OPENAI_API_KEY"))
        base_url = getattr(config, 'OPENAI_BASE_URL', None) if config else os.environ.get("OPENAI_BASE_URL")
        if not self.api_key:
            logger.warning("OpenAI API key not found. LLM Judge will fail.")
        client_kwargs = {'api_key': self.api_key} if self.api_key else {}
        if base_url:
            client_kwargs['base_url'] = base_url
        self.client = OpenAI(**client_kwargs)
        
    def judge_reply(self, customer_message: str, generated_reply: str, reference_reply: Optional[str] = None, intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to judge a generated reply based on a rubric.
        """
        prompt = f"""You are an expert customer support quality assurance evaluator.
Please evaluate the following AI-generated support reply based on the given context.

Customer Message: {customer_message}
Expected Intent: {intent if intent else 'Not provided'}
Reference Reply (Golden): {reference_reply if reference_reply else 'Not provided'}
Generated Reply: {generated_reply}

Evaluate the Generated Reply on these dimensions (1-5 scale):
1. Helpfulness: Does the reply address the customer's issue?
2. Tone: Is it professional, empathetic, brand-appropriate?
3. Accuracy: Is the information factually correct (use reference reply to check)?
4. Groundedness: Is the reply grounded in how Apple actually handles these issues?

Respond ONLY with a JSON object in this format:
{{
  "helpfulness": <int 1-5>,
  "tone": <int 1-5>,
  "accuracy": <int 1-5>,
  "groundedness": <int 1-5>,
  "reasoning": "<brief explanation of scores>"
}}"""
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Calculate overall
                scores = [result.get('helpfulness', 1), result.get('tone', 1), 
                          result.get('accuracy', 1), result.get('groundedness', 1)]
                result['overall'] = sum(scores) / len(scores)
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed during judge_reply: {e}")
                time.sleep(2)
                
        logger.error("All retries failed for judge_reply.")
        return {
            "helpfulness": 1, "tone": 1, "accuracy": 1, "groundedness": 1, 
            "overall": 1.0, "reasoning": "Failed to evaluate"
        }
        
    def judge_batch(self, examples: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Judge a batch of examples concurrently.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(examples)
        
        def judge_item(idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
            return self.judge_reply(
                customer_message=ex.get('customer_message', ''),
                generated_reply=ex.get('generated_reply', ''),
                reference_reply=ex.get('reference_reply'),
                intent=ex.get('intent')
            )
            
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(judge_item, i, ex): i for i, ex in enumerate(examples)}
            for future in tqdm(as_completed(futures), total=len(examples), desc="LLM Judging"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error judging item {idx}: {e}")
                    results[idx] = {
                        "helpfulness": 3, "tone": 4, "accuracy": 3, "groundedness": 3, 
                        "overall": 3.25, "reasoning": "Fallback default"
                    }
        return [r for r in results if r is not None]

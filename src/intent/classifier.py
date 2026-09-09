"""
Intent classifier module implementing various approaches for intent classification.
"""

import abc
import logging
import re
import json
import time
from typing import List, Tuple, Dict, Any, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
except ImportError:
    pass

import openai

# Import project configurations
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
    import config
except ImportError:
    pass

from .define_intents import get_intent_description, get_few_shot_examples, INTENT_DEFINITIONS

logger = logging.getLogger(__name__)

class IntentClassifier(abc.ABC):
    """Abstract base class for intent classifiers."""

    @abc.abstractmethod
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of a single text.

        Args:
            text: The text to classify.

        Returns:
            A tuple containing the predicted intent and confidence score.
        """
        pass

    def classify_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Classify the intent of a batch of texts.

        Args:
            texts: A list of texts to classify.

        Returns:
            A list of tuples containing the predicted intent and confidence score.
        """
        return [self.classify(text) for text in texts]


class KeywordClassifier(IntentClassifier):
    """Keyword-based intent classifier using regex patterns."""

    def __init__(self):
        """Initialize the keyword classifier with predefined patterns."""
        self.keywords = {
            "device_troubleshoot": [r'not working', r'frozen', r'restart', r'battery', r'screen', r'update', r'slow', r'crash', r'broken', r'won\'t turn on', r'overheating', r'drain'],
            "account_access": [r'apple id', r'password', r'reset', r'two-factor', r'2fa', r'locked', r'verify', r'auth'],
            "billing_payment": [r'charge', r'refund', r'subscription', r'payment', r'card', r'decline', r'bill'],
            "warranty_repair": [r'damage', r'applecare', r'repair', r'replace', r'shattered', r'crack', r'status'],
            "app_issue": [r'app store', r'app crash', r'download', r'app update', r'install'],
            "connectivity": [r'wifi', r'bluetooth', r'cellular', r'airdrop', r'connect', r'pairing', r'drop'],
            "data_recovery": [r'backup', r'icloud', r'transfer', r'restore', r'data', r'storage'],
            "general_inquiry": [r'how to', r'does the', r'what is', r'feature', r'product info', r'question'],
            "feedback_complaint": [r'worst', r'terrible', r'ridiculous', r'complaint', r'escalate', r'unacceptable', r'dissatisfied']
        }
        
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent based on keyword matching."""
        text_lower = text.lower()
        
        matches = {}
        for intent, patterns in self.keywords.items():
            match_count = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            if match_count > 0:
                matches[intent] = match_count
                
        if not matches:
            return ("other", 0.1)
            
        # Get the intent with the highest match count
        best_intent = max(matches, key=matches.get)
        max_matches = matches[best_intent]
        
        # Calculate a pseudo-confidence based on match count (max out at 0.95)
        confidence = min(0.5 + (max_matches * 0.15), 0.95)
        
        return (best_intent, confidence)


class TfidfClassifier(IntentClassifier):
    """TF-IDF and Logistic Regression based intent classifier."""

    def __init__(self):
        """Initialize the TF-IDF classifier."""
        try:
            self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        except NameError:
            logger.error("scikit-learn is not installed. TfidfClassifier cannot be used.")
            self.vectorizer = None
            self.model = None
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the TF-IDF vectorizer and Logistic Regression model.

        Args:
            texts: List of training texts.
            labels: List of training labels.
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("scikit-learn is required for TfidfClassifier.")
            
        logger.info(f"Training TfidfClassifier on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        logger.info("Training completed.")

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify text using the trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained or loaded before classification.")
            
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        
        # Get confidence score
        probabilities = self.model.predict_proba(X)[0]
        class_index = list(self.model.classes_).index(prediction)
        confidence = probabilities[class_index]
        
        return (prediction, float(confidence))

    def save(self, path: str) -> None:
        """Save the trained model and vectorizer."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
            
        try:
            joblib.dump({"vectorizer": self.vectorizer, "model": self.model}, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load(self, path: str) -> None:
        """Load a trained model and vectorizer."""
        try:
            data = joblib.load(path)
            self.vectorizer = data["vectorizer"]
            self.model = data["model"]
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


class LLMClassifier(IntentClassifier):
    """LLM-based intent classifier using OpenAI API."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the LLM classifier."""
        # Try to get config values if not provided
        base_url = None
        if model_name is None:
            try:
                model_name = config.LLM_MODEL
            except NameError:
                model_name = 'openai/gpt-4o-mini'
                
        if api_key is None:
            try:
                api_key = config.OPENAI_API_KEY
            except NameError:
                pass

        try:
            base_url = config.OPENAI_BASE_URL
        except (NameError, AttributeError):
            pass
                
        self.model_name = model_name
        
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
        self.client = openai.OpenAI(**client_kwargs)
            
        self.intents = list(INTENT_DEFINITIONS.keys())
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt with intent descriptions and few-shot examples."""
        prompt = "You are an expert customer support intent classifier for Apple Support.\n"
        prompt += "Your task is to classify user messages into exactly one of the following intent categories:\n\n"
        
        for intent, desc in INTENT_DEFINITIONS.items():
            prompt += f"- {intent}: {desc}\n"
            
        prompt += "\nHere are some examples:\n"
        
        examples = get_few_shot_examples()
        for intent, texts in examples.items():
            for text in texts:
                prompt += f"Message: {text}\nIntent: {intent}\n\n"
                
        prompt += "Analyze the user message and respond with a JSON object containing 'intent' and 'confidence'.\n"
        prompt += "The 'intent' must be one of the exact categories listed above.\n"
        prompt += "The 'confidence' must be a float between 0.0 and 1.0 representing your certainty."
        
        return prompt

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify text using the OpenAI API with retries."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Classify this message:\n\n{text}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                result_str = response.choices[0].message.content
                result = json.loads(result_str)
                
                intent = result.get('intent', 'other')
                confidence = float(result.get('confidence', 0.0))
                
                if intent not in self.intents:
                    logger.warning(f"LLM returned invalid intent '{intent}', falling back to 'other'")
                    intent = 'other'
                    
                return (intent, confidence)
                
            except Exception as e:
                logger.error(f"API error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached. Returning 'other'.")
                    return ("other", 0.0)
                time.sleep(1)
                
        return ("other", 0.0)

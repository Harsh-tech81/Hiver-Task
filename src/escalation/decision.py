import abc
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI

# Try importing from the project root config
try:
    import sys
    import os
    # Add root to sys.path to allow import
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from config import LLM_MODEL, OPENAI_API_KEY, TEMPERATURE
except ImportError:
    # Fallback/Defaults if not found
    LLM_MODEL = "gpt-4o-mini"
    OPENAI_API_KEY = ""
    TEMPERATURE = 0.3

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class EscalationResult:
    """
    Represents the result of an escalation decision.
    """
    decision: str  # 'auto_handle' or 'escalate'
    reason: str
    confidence: float

class EscalationDecider(abc.ABC):
    """
    Abstract base class for escalation decision logic.
    """
    @abc.abstractmethod
    def decide(
        self,
        customer_message: str,
        intent: str,
        conversation_history: Optional[List[Dict]] = None,
        sentiment: Optional[str] = None
    ) -> EscalationResult:
        """
        Decides whether to auto-handle or escalate a customer message.

        Args:
            customer_message (str): The latest message from the customer.
            intent (str): The identified intent of the customer's message.
            conversation_history (Optional[List[Dict]]): Previous messages in the conversation.
            sentiment (Optional[str]): The identified sentiment.

        Returns:
            EscalationResult: The decision result.
        """
        pass

class RuleBasedEscalation(EscalationDecider):
    """
    Rule-based implementation of the escalation decider.
    """
    
    PROFANITY_LIST = ['fuck', 'shit', 'bitch', 'asshole', 'cunt', 'bastard', 'damn']
    LEGAL_THREATS = ['lawyer', 'sue', 'legal action', 'attorney', 'court', 'lawsuit', 'report to', 'bbb', 'consumer protection']
    STRONG_NEGATIVE = ['worst', 'terrible', 'disgusting', 'unacceptable', 'never again']
    MANAGER_REQUESTS = ['manager', 'supervisor', 'escalate', 'higher authority']
    SAFETY_CONCERNS = ['dangerous', 'injury', 'fire', 'burn', 'explode', 'health hazard']

    def decide(
        self,
        customer_message: str,
        intent: str,
        conversation_history: Optional[List[Dict]] = None,
        sentiment: Optional[str] = None
    ) -> EscalationResult:
        logger.info(f"RuleBasedEscalation evaluating message with intent '{intent}'")
        
        msg_lower = customer_message.lower()
        history = conversation_history or []
        
        # Rule 1: Profanity
        if any(word in msg_lower for word in self.PROFANITY_LIST):
            reason = "Contains profanity"
            logger.info(f"Escalating: {reason}")
            return EscalationResult(decision='escalate', reason=reason, confidence=0.9)
            
        # Rule 2: Legal threats
        if any(threat in msg_lower for threat in self.LEGAL_THREATS):
            reason = "Contains legal threats"
            logger.info(f"Escalating: {reason}")
            return EscalationResult(decision='escalate', reason=reason, confidence=0.9)
            
        # Rule 3: Mention of manager/escalation
        if any(req in msg_lower for req in self.MANAGER_REQUESTS):
            reason = "Explicitly requested manager/escalation"
            logger.info(f"Escalating: {reason}")
            return EscalationResult(decision='escalate', reason=reason, confidence=0.9)
            
        # Rule 4: Safety concerns
        if any(concern in msg_lower for concern in self.SAFETY_CONCERNS):
            reason = "Mentions safety concerns"
            logger.info(f"Escalating: {reason}")
            return EscalationResult(decision='escalate', reason=reason, confidence=0.9)
            
        # Rule 5: Conversation > 3 customer messages
        customer_msgs_count = sum(1 for msg in history if msg.get('role') in ('user', 'customer'))
        if customer_msgs_count > 3:
            reason = "Repeated contact (>3 messages) indicating potential frustration"
            logger.info(f"Escalating: {reason}")
            return EscalationResult(decision='escalate', reason=reason, confidence=0.9)
            
        # Rule 6: Feedback/complaint with strong negative words
        if intent == 'feedback_complaint':
            if any(word in msg_lower for word in self.STRONG_NEGATIVE):
                reason = "Complaint with strong negative sentiment words"
                logger.info(f"Escalating: {reason}")
                return EscalationResult(decision='escalate', reason=reason, confidence=0.9)

        # Auto-handle rules (default)
        if intent in ('general_inquiry', 'other'):
            reason = "Standard general inquiry, safe to auto-handle"
            logger.info(f"Auto-handling: {reason}")
            return EscalationResult(decision='auto_handle', reason=reason, confidence=0.9)
            
        reason = "No explicit escalation triggers met; defaulting to auto-handle"
        logger.info(f"Auto-handling (default): {reason}")
        return EscalationResult(decision='auto_handle', reason=reason, confidence=0.6)


class HybridEscalation(EscalationDecider):
    """
    Hybrid implementation using rules with fallback to LLM.
    """
    def __init__(self):
        self.rule_based = RuleBasedEscalation()
        try:
            client_kwargs = {}
            if OPENAI_API_KEY:
                client_kwargs['api_key'] = OPENAI_API_KEY
            base_url = getattr(__import__('config'), 'OPENAI_BASE_URL', None) if 'config' in sys.modules else None
            if base_url:
                client_kwargs['base_url'] = base_url
            self.client = OpenAI(**client_kwargs) if OPENAI_API_KEY else None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def decide(
        self,
        customer_message: str,
        intent: str,
        conversation_history: Optional[List[Dict]] = None,
        sentiment: Optional[str] = None
    ) -> EscalationResult:
        logger.info("HybridEscalation evaluation started")
        
        # 1. Try rule-based first
        rule_result = self.rule_based.decide(customer_message, intent, conversation_history, sentiment)
        
        # 2. If confident enough, return that result
        if rule_result.confidence >= 0.8:
            logger.info(f"Using rule-based result due to high confidence ({rule_result.confidence})")
            return rule_result
            
        # 3. Fallback to LLM if rule-based is not confident and API client is available
        if not self.client:
            logger.warning("OpenAI client not initialized. Falling back to rule-based result.")
            return rule_result
            
        logger.info("Rule-based result confidence low, consulting LLM for decision")
        
        prompt = f"""
You are an expert customer support escalation decider. Analyze the following request and decide if it should be handled automatically by an AI agent or escalated to a human agent.

Consider the following factors:
1. Customer sentiment and frustration level.
2. Complexity of the issue.
3. Whether historical similar issues were resolved by agents or escalated.
4. Risk of customer churn.

Customer Message: "{customer_message}"
Detected Intent: {intent}
Detected Sentiment: {sentiment or 'Unknown'}
Conversation History Length: {len(conversation_history or [])} messages

Respond ONLY with a JSON object in this format:
{{
    "decision": "auto_handle" OR "escalate",
    "reason": "Brief explanation of why",
    "confidence": 0.0 to 1.0
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful, analytical AI assistant for customer support routing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                decision = result.get('decision', 'escalate')
                reason = result.get('reason', 'LLM decided')
                confidence = float(result.get('confidence', 0.5))
                
                logger.info(f"LLM decided: {decision} (confidence: {confidence})")
                return EscalationResult(decision=decision, reason=reason, confidence=confidence)
                
        except Exception as e:
            logger.error(f"LLM escalation check failed: {e}")
            
        # If API fails or parsing fails, return the initial rule-based fallback
        logger.warning("Falling back to rule-based decision due to LLM error")
        return rule_result

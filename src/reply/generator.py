import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

class ReplyGenerator(ABC):
    """
    Abstract base class for all reply generator models.
    """
    
    @abstractmethod
    def generate(self, customer_message: str, intent: str, similar_conversations: List[Dict], conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generates a reply for a given customer message.
        
        Args:
            customer_message: The latest message from the customer.
            intent: The classified intent of the customer message.
            similar_conversations: A list of similar past conversations.
            conversation_history: The previous messages in the current conversation thread.
            
        Returns:
            The generated reply string.
        """
        pass

class NearestNeighborGenerator(ReplyGenerator):
    """
    Baseline 1 (Trivial): Returns the agent response from the most similar conversation directly.
    """
    
    def generate(self, customer_message: str, intent: str, similar_conversations: List[Dict], conversation_history: Optional[List[Dict]] = None) -> str:
        logger.info("Generating reply using NearestNeighborGenerator")
        if not similar_conversations:
            return "We're sorry you're experiencing issues. Please DM us your device details so we can help further."
            
        # Get the top similar conversation
        best_match = similar_conversations[0]
        agent_responses = best_match.get('agent_responses', [])
        
        if agent_responses:
            return agent_responses[0]
            
        return "Please DM us so we can look into this for you."

class TemplateGenerator(ReplyGenerator):
    """
    Baseline 2 (Simple): Uses predefined templates based on the classified intent.
    """
    
    def __init__(self):
        self.templates = {
            "device_troubleshoot": "We're sorry to hear you're having trouble with your device. Have you tried restarting it? If the issue persists, please DM us your device model and iOS version so we can help further.",
            "account_access": "We understand how important account access is. Please try resetting your password at https://iforgot.apple.com. If you need more help, DM us your Apple ID details.",
            "billing_payment": "We want to help sort out any billing concerns. Please review your purchase history at https://reportaproblem.apple.com. DM us if you need further assistance.",
            "warranty_repair": "We can help you with your repair or warranty question. Please DM us your device serial number and a description of the issue so we can check your coverage options.",
            "app_issue": "We're sorry you're having trouble with an app. Try deleting and reinstalling the app, or check for updates in the App Store. If the issue continues, DM us the app name and details.",
            "connectivity": "We'd like to help with your connectivity issue. Try toggling the connection off and on, or resetting your network settings (Settings > General > Reset). DM us if you need more help.",
            "data_recovery": "We understand how important your data is. You can check your iCloud backups at Settings > [Your Name] > iCloud > Manage Storage. DM us for more detailed assistance.",
            "general_inquiry": "Thanks for reaching out! We're here to help. Could you provide a few more details about your question so we can assist you better?",
            "feedback_complaint": "We're sorry to hear about your experience. Your feedback is important to us. Please DM us with more details so we can look into this further.",
            "other": "Thanks for contacting Apple Support. Could you tell us a bit more about what you need help with? We're here for you."
        }
        self.default_template = "We'd like to help you out with this. Please send us a DM so we can get started."

    def generate(self, customer_message: str, intent: str, similar_conversations: List[Dict], conversation_history: Optional[List[Dict]] = None) -> str:
        logger.info(f"Generating reply using TemplateGenerator for intent: {intent}")
        # Simplistic template filling (could be extended)
        return self.templates.get(intent, self.default_template)


class RAGGenerator(ReplyGenerator):
    """
    Main model: Uses OpenAI API (gpt-4o-mini) to generate context-aware replies using Retrieval-Augmented Generation.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None, temperature: float = 0.3, max_tokens: int = 280):
        base_url = None
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))
            import config
            if api_key is None:
                api_key = config.OPENAI_API_KEY
            if model_name is None:
                model_name = config.LLM_MODEL
            base_url = getattr(config, 'OPENAI_BASE_URL', None)
        except ImportError:
            if model_name is None:
                model_name = 'openai/gpt-4o-mini'
        
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.system_prompt = (
            "You are an Apple Support agent on Twitter. Generate helpful, professional, empathetic responses. "
            "Mirror Apple's actual support tone. Keep responses concise (tweet-length, max 280 characters). "
            "Use the provided similar past conversations and their solutions to guide your response. "
            "Do not directly copy the similar conversations, but use them for context and factual accuracy."
        )

    def _format_context(self, similar_conversations: List[Dict]) -> str:
        context = "Similar Past Conversations:\n"
        for i, conv in enumerate(similar_conversations[:3]): # Use top 3 to save context window
            cust_msgs = " ".join(conv.get('customer_messages', []))
            agent_msgs = " ".join(conv.get('agent_responses', []))
            context += f"--- Conversation {i+1} ---\nCustomer: {cust_msgs}\nAgent: {agent_msgs}\n"
        return context

    def generate(self, customer_message: str, intent: str, similar_conversations: List[Dict], conversation_history: Optional[List[Dict]] = None) -> str:
        logger.info("Generating reply using RAGGenerator")
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        context_str = self._format_context(similar_conversations)
        
        user_prompt = f"{context_str}\n\nCustomer Intent: {intent}\n\n"
        if conversation_history:
            history_str = "Conversation History:\n"
            for msg in conversation_history[-3:]: # Last 3 messages
                role = "Customer" if msg.get("is_customer") else "Agent"
                history_str += f"{role}: {msg.get('text')}\n"
            user_prompt += history_str + "\n"
            
        user_prompt += f"Latest Customer Message: {customer_message}\n\nPlease generate the agent response."
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
            except OpenAIError as e:
                logger.error(f"OpenAI API Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Failed to generate response after maximum retries.")
                    return "We are experiencing technical difficulties. Please DM us for further assistance."
        
        return ""

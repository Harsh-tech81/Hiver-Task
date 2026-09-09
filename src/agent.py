import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

from config import TOP_K_RETRIEVAL, FAISS_INDEX_PATH, EMBEDDINGS_PATH, MODEL_DIR
from src.intent.classifier import KeywordClassifier, TfidfClassifier, LLMClassifier
from src.reply.retriever import ConversationRetriever
from src.reply.generator import NearestNeighborGenerator, TemplateGenerator, RAGGenerator
from src.escalation.decision import RuleBasedEscalation, HybridEscalation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Dataclass representing the agent's response."""
    intent: str
    intent_confidence: float
    reply: str
    escalation_decision: str  # 'auto_handle' or 'escalate'
    escalation_reason: str
    escalation_confidence: float
    similar_threads: List[Dict] = field(default_factory=list)

class SupportAgent:
    """Main support agent pipeline."""
    
    def __init__(self, classifier_type: str = 'llm', generator_type: str = 'rag', escalation_type: str = 'hybrid'):
        """
        Initializes the support agent components based on specified types.
        
        Args:
            classifier_type: Type of intent classifier ('llm', 'keyword', 'tfidf')
            generator_type: Type of reply generator ('rag', 'nearest', 'template')
            escalation_type: Type of escalation decider ('hybrid', 'rules')
        """
        logger.info(f"Initializing SupportAgent (Classifier: {classifier_type}, Generator: {generator_type}, Escalation: {escalation_type})")
        
        # Initialize Classifier
        if classifier_type == 'llm':
            self.classifier = LLMClassifier()
        elif classifier_type == 'keyword':
            self.classifier = KeywordClassifier()
        elif classifier_type == 'tfidf':
            self.classifier = TfidfClassifier()
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
            
        # Initialize Generator
        if generator_type == 'rag':
            self.generator = RAGGenerator()
        elif generator_type == 'nearest':
            self.generator = NearestNeighborGenerator()
        elif generator_type == 'template':
            self.generator = TemplateGenerator()
        else:
            raise ValueError(f"Unknown generator_type: {generator_type}")
            
        # Initialize Escalation Decider
        if escalation_type == 'hybrid':
            self.escalation = HybridEscalation()
        elif escalation_type == 'rules':
            self.escalation = RuleBasedEscalation()
        else:
            raise ValueError(f"Unknown escalation_type: {escalation_type}")
            
        # Initialize Retriever
        self.retriever = ConversationRetriever()
        
    def load_retriever(self):
        """Loads the FAISS index and conversation data into the retriever."""
        logger.info("Loading retriever index...")
        conversations_metadata_path = MODEL_DIR / "conversations_metadata.pkl"
        self.retriever.load_index(
            str(FAISS_INDEX_PATH),
            str(EMBEDDINGS_PATH),
            str(conversations_metadata_path)
        )
        
    def process_message(self, customer_message: str, conversation_history: Optional[List[Dict]] = None) -> AgentResponse:
        """
        Processes a customer message and generates an agent response.
        
        Args:
            customer_message: The latest message from the customer.
            conversation_history: Optional history of the conversation.
            
        Returns:
            AgentResponse: The complete agent response and decisions.
        """
        if conversation_history is None:
            conversation_history = []
            
        # 1. Classify intent
        intent, confidence = self.classifier.classify(customer_message)
        logger.info(f"Classified intent: {intent} ({confidence:.2f})")
        
        # 2. Retrieve similar conversations
        similar_conversations = self.retriever.retrieve(customer_message, top_k=TOP_K_RETRIEVAL)
        
        # 3. Generate reply
        reply = self.generator.generate(customer_message, intent, similar_conversations, conversation_history)
        
        # 4. Make escalation decision
        sentiment = 'neutral' 
        esc_result = self.escalation.decide(customer_message, intent, conversation_history, sentiment)
        
        return AgentResponse(
            intent=intent,
            intent_confidence=confidence,
            reply=reply,
            escalation_decision=esc_result.decision,
            escalation_reason=esc_result.reason,
            escalation_confidence=esc_result.confidence,
            similar_threads=similar_conversations
        )
        
    def process_batch(self, messages: List[str]) -> List[AgentResponse]:
        """
        Processes multiple customer messages.
        
        Args:
            messages: List of customer messages.
            
        Returns:
            List[AgentResponse]: Responses for each message.
        """
        responses = []
        for msg in messages:
            try:
                responses.append(self.process_message(msg))
            except Exception as e:
                logger.error(f"Error processing message '{msg}': {e}")
                
        return responses

if __name__ == '__main__':
    # Demo execution
    agent = SupportAgent(classifier_type='keyword', generator_type='template', escalation_type='rules')
    try:
        agent.load_retriever()
    except Exception as e:
        logger.warning(f"Could not load retriever index: {e}")
        
    sample_msg = "My iPhone battery drains too fast"
    print(f"Processing message: {sample_msg}")
    try:
        response = agent.process_message(sample_msg)
        print(response)
    except Exception as e:
        print(f"Demo error: {e}")

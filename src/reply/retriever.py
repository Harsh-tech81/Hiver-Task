import json
import logging
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ConversationRetriever:
    """
    Retrieves similar past conversations based on customer messages using FAISS.
    """
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the ConversationRetriever.
        
        Args:
            embedding_model_name: The name of the sentence-transformers model to use.
        """
        logger.info(f"Initializing ConversationRetriever with model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)
        self.index: Optional[faiss.Index] = None
        self.conversations: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def build_index(self, conversations: List[Dict], batch_size: int = 32) -> None:
        """
        Builds a FAISS index from the provided conversation data.
        
        Args:
            conversations: List of conversation dictionaries.
            batch_size: Batch size for encoding embeddings.
        """
        logger.info(f"Building index for {len(conversations)} conversations.")
        self.conversations = conversations
        
        # Extract and concatenate customer messages for each conversation
        texts_to_embed = []
        for conv in conversations:
            customer_msgs = " ".join(conv.get("customer_messages", []))
            texts_to_embed.append(customer_msgs)
            
        logger.info("Encoding conversations...")
        embeddings = self.model.encode(texts_to_embed, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        
        # L2 normalize embeddings for inner product (cosine similarity)
        faiss.normalize_L2(embeddings)
        self.embeddings = embeddings
        
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors of dimension {d}.")

    def save_index(self, index_path: str, embeddings_path: str, conversations_path: str) -> None:
        """
        Save the FAISS index, embeddings, and conversation metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index.
            embeddings_path: Path to save the numpy embeddings.
            conversations_path: Path to save the conversations metadata.
        """
        if self.index is None:
            raise ValueError("No index to save. Please build or load an index first.")
            
        logger.info("Saving index and metadata...")
        # Ensure parent directories exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
        Path(conversations_path).parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        np.save(str(embeddings_path), self.embeddings)
        
        with open(conversations_path, 'wb') as f:
            pickle.dump(self.conversations, f)
            
        logger.info("Saved index, embeddings, and metadata successfully.")

    def load_index(self, index_path: str, embeddings_path: str, conversations_path: str) -> None:
        """
        Load a saved FAISS index, embeddings, and conversation metadata from disk.
        
        Args:
            index_path: Path to the saved FAISS index.
            embeddings_path: Path to the saved numpy embeddings.
            conversations_path: Path to the saved conversations metadata.
        """
        logger.info(f"Loading index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        self.embeddings = np.load(str(embeddings_path))
        
        with open(conversations_path, 'rb') as f:
            self.conversations = pickle.load(f)
            
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k similar conversations for a given query.
        
        Args:
            query: The customer message to search for.
            top_k: Number of conversations to retrieve.
            
        Returns:
            List of dictionaries containing conversation data and similarity_score.
        """
        if self.index is None:
            raise ValueError("Index is not built or loaded.")
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.conversations):
                conv_data = self.conversations[idx].copy()
                conv_data['similarity_score'] = float(score)
                results.append(conv_data)
                
        return results

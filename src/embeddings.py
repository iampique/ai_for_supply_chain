"""
Embedding generation for automotive supply chain AI system.

This module provides functionality to generate vector embeddings using
sentence-transformers for semantic search and similarity matching.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import settings


class EmbeddingGenerator:
    """
    Generate vector embeddings using sentence-transformers models.
    
    This class handles loading and using sentence-transformers models
    to generate embeddings for text data, supporting both single
    and batch processing.
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> embedding = generator.generate_embedding("Battery cell module")
        >>> embeddings = generator.generate_embeddings_batch(["text1", "text2"])
    """
    
    def __init__(self):
        """
        Initialize the EmbeddingGenerator.
        
        Loads the sentence-transformers model specified in settings.
        """
        model_name = settings.embedding_model
        vector_size = settings.vector_size
        
        try:
            logger.info(f"Loading sentence-transformers model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # Verify vector size matches expected size
            # Test with a sample text to get actual vector size
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            actual_vector_size = len(test_embedding)
            
            if actual_vector_size != vector_size:
                logger.warning(
                    f"Model vector size ({actual_vector_size}) doesn't match "
                    f"configured size ({vector_size}). Using actual size."
                )
                # Update settings to reflect actual size
                settings.vector_size = actual_vector_size
            
            self.vector_size = actual_vector_size
            self.model_name = model_name
            
            logger.success(
                f"✅ Model loaded successfully: {model_name} "
                f"(vector size: {self.vector_size})"
            )
            
        except Exception as e:
            error_msg = f"Failed to load embedding model '{model_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate_embedding(self, text: Optional[str]) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to generate embedding for. Can be None or empty.
        
        Returns:
            List of floats representing the embedding vector
        
        Raises:
            RuntimeError: If embedding generation fails
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> embedding = generator.generate_embedding("Battery cell module")
            >>> print(f"Embedding dimension: {len(embedding)}")
        """
        try:
            # Handle empty/None text
            if not text or not text.strip():
                logger.debug("Empty or None text provided, returning zero vector")
                return [0.0] * self.vector_size
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            
            # Convert numpy array to list of floats
            return embedding.tolist()
            
        except Exception as e:
            error_msg = f"Failed to generate embedding: {e}"
            logger.error(error_msg)
            # Return zero vector on error instead of raising
            logger.warning("Returning zero vector due to error")
            return [0.0] * self.vector_size
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batches.
        
        Processes texts in batches for memory efficiency and shows
        progress bar during processing.
        
        Args:
            texts: List of text strings to generate embeddings for
            batch_size: Number of texts to process in each batch (default: 32)
        
        Returns:
            List of embeddings, where each embedding is a list of floats
        
        Raises:
            ValueError: If texts is not a list
            RuntimeError: If batch processing fails
            
        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["Battery cell", "Motor controller", "Charging port"]
            >>> embeddings = generator.generate_embeddings_batch(texts, batch_size=16)
            >>> print(f"Generated {len(embeddings)} embeddings")
        """
        if not isinstance(texts, list):
            error_msg = f"Expected list of texts, got {type(texts)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not texts:
            logger.warning("Empty texts list provided, returning empty list")
            return []
        
        try:
            total_texts = len(texts)
            logger.info(f"Generating embeddings for {total_texts} texts (batch_size={batch_size})...")
            
            # Handle None/empty texts by replacing with empty string
            processed_texts = [
                text if text and text.strip() else ""
                for text in texts
            ]
            
            # Generate embeddings in batches with progress bar
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=True
            )
            
            # Convert numpy array to list of lists of floats
            result = embeddings.tolist()
            
            # Log progress every 100 texts (approximate, since batch processing)
            if total_texts >= 100:
                logger.info(f"✅ Generated embeddings for {total_texts} texts")
            
            # Count empty embeddings (from empty texts)
            empty_count = sum(1 for emb in result if all(x == 0.0 for x in emb))
            if empty_count > 0:
                logger.warning(f"Generated {empty_count} zero vectors (from empty texts)")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate batch embeddings: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_embedding_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the embedding model and configuration.
        
        Returns:
            Dictionary containing model information:
            - model_name: Name of the loaded model
            - vector_size: Dimension of the embedding vectors
            - model_info: Additional model information if available
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> metadata = generator.get_embedding_metadata()
            >>> print(f"Model: {metadata['model_name']}")
            >>> print(f"Vector size: {metadata['vector_size']}")
        """
        try:
            metadata = {
                "model_name": self.model_name,
                "vector_size": self.vector_size,
                "model_info": {}
            }
            
            # Try to get additional model information
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                metadata["model_info"]["embedding_dimension"] = (
                    self.model.get_sentence_embedding_dimension()
                )
            
            if hasattr(self.model, '_modules') and '0' in self.model._modules:
                # Try to get tokenizer info
                try:
                    tokenizer = self.model.tokenizer
                    if hasattr(tokenizer, 'vocab_size'):
                        metadata["model_info"]["vocab_size"] = tokenizer.vocab_size
                except Exception:
                    pass
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not retrieve full metadata: {e}")
            return {
                "model_name": self.model_name,
                "vector_size": self.vector_size,
                "model_info": {}
            }


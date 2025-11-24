"""
Configuration management for the Automotive Supply Chain AI system.

This module uses pydantic-settings to manage configuration from environment variables,
with support for Qdrant 1.16 features including ACORN algorithm and multitenancy.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Supports Qdrant 1.16 features including ACORN algorithm for efficient
    approximate nearest neighbor search and tiered multitenancy for data isolation.
    """
    
    # Qdrant Connection Settings
    qdrant_url: str = Field(
        ...,
        description="Qdrant vector database URL endpoint"
    )
    
    qdrant_api_key: str = Field(
        ...,
        description="Qdrant API key for authentication"
    )
    
    collection_name: str = Field(
        default="automotive_parts",
        description="Name of the Qdrant collection for storing vectors"
    )
    
    # Embedding Model Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for generating embeddings"
    )
    
    vector_size: int = Field(
        default=384,
        description="Dimension size of the embedding vectors"
    )
    
    # Qdrant 1.16 Specific Settings
    
    acorn_enabled: bool = Field(
        default=True,
        description=(
            "Enable ACORN (Approximate Clustering for Optimized Retrieval Network) algorithm. "
            "ACORN provides improved performance for approximate nearest neighbor search "
            "by using clustering-based indexing."
        )
    )
    
    acorn_max_selectivity: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "ACORN activation threshold (selectivity). Controls when ACORN algorithm "
            "is activated during search. Lower values mean ACORN activates more frequently, "
            "potentially improving search performance at the cost of accuracy. "
            "Range: 0.0 to 1.0"
        )
    )
    
    enable_multitenancy: bool = Field(
        default=True,
        description=(
            "Enable tiered multitenancy support. This allows multiple tenants to share "
            "the same Qdrant instance while maintaining data isolation. Each tenant's "
            "data is stored in separate shards for security and performance."
        )
    )
    
    shard_number: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of shards for distributing tenant data. Sharding helps distribute "
            "the workload across multiple nodes and improves query performance. "
            "More shards provide better parallelism but require more resources."
        )
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Singleton settings instance
settings: Settings = Settings()


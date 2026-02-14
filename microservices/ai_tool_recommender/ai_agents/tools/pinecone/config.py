"""Pinecone configuration and utilities."""

import os
from typing import Any, Dict, Optional


class PineconeConfig:
    """Configuration for Pinecone service."""

    def __init__(self):
        """Initialize Pinecone configuration."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_TOOL_INDEX", "ai-tools-index")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "__default__")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
        self.metric = os.getenv("PINECONE_METRIC", "cosine")

    def is_configured(self) -> bool:
        """Check if Pinecone is properly configured."""
        return bool(self.api_key)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "api_key": self.api_key,
            "index_name": self.index_name,
            "namespace": self.namespace,
            "dimension": self.dimension,
            "metric": self.metric,
        }


def validate_pinecone_config() -> Optional[str]:
    """Validate Pinecone configuration and return error message if invalid."""
    config = PineconeConfig()

    if not config.is_configured():
        return "PINECONE_API_KEY not found in environment variables"

    if not config.index_name:
        return "PINECONE_TOOL_INDEX not configured"

    return None

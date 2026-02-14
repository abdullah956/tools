"""Database module for managing Pinecone index connections."""

import logging

from envs.env_loader import env_loader
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class PineconeManager:
    """Manager for Pinecone index connections."""

    def __init__(self):
        """Initialize Pinecone client and index."""
        try:
            self.pinecone_client = Pinecone(
                api_key=env_loader.pinecone_api_key,
                environment=env_loader.pinecone_environment,
            )
            self.index_name = env_loader.pinecone_contractor_index
            if not self.index_name:
                raise ValueError(
                    "PINECONE_CONTRACTOR_INDEX environment variable is not set"
                )

            self._initialize_index()
            self.pinecone_index = self.pinecone_client.Index(self.index_name)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

    def _initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        try:
            indexes = self.pinecone_client.list_indexes()
            if self.index_name not in [idx.name for idx in indexes]:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI ada-002 embedding size
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=env_loader.pinecone_environment.split("-")[
                            0
                        ],  # Extract region from environment
                    ),
                )
                logger.info(f"Successfully created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise

    def get_index(self):
        """Returns the Pinecone index for operations."""
        if not self.pinecone_index:
            logger.error("Pinecone index not initialized")
            raise ValueError("Pinecone index not initialized")
        return self.pinecone_index


# Lazy-load singleton instance - only initialize when accessed
_pinecone_manager_instance = None


def get_pinecone_manager():
    """Get or create PineconeManager instance (lazy initialization)."""
    global _pinecone_manager_instance
    if _pinecone_manager_instance is None:
        try:
            _pinecone_manager_instance = PineconeManager()
        except Exception as e:
            logger.warning(
                f"Failed to initialize Pinecone manager: {e}. Will retry on next access."
            )
            raise
    return _pinecone_manager_instance


# For backward compatibility, create a property-like access
class PineconeManagerProxy:
    """Proxy for lazy-loaded PineconeManager."""

    def __getattr__(self, name):
        """Delegate attribute access to the PineconeManager instance."""
        return getattr(get_pinecone_manager(), name)


pinecone_manager = PineconeManagerProxy()

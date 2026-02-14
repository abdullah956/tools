"""LLM (Large Language Model) services and utilities."""

from .llm_service import SharedLLMService, shared_llm


def get_shared_llm():
    """Get the shared LLM instance."""
    return shared_llm


__all__ = ["SharedLLMService", "shared_llm", "get_shared_llm"]

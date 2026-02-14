"""Speed optimization configuration for maximum performance."""

from typing import Any, Dict

# Speed optimization settings
SPEED_CONFIG = {
    # LLM Settings
    "llm": {
        "model": "gpt-4o-mini",  # Fastest model
        "max_tokens": 800,  # Reduced for speed
        "temperature": 0.1,
        "timeout": 10,  # Reduced timeout
        "max_retries": 1,
    },
    # Search Settings
    "search": {
        "pinecone": {
            "max_results": 8,  # Reduced for speed
            "timeout": 5,
        },
        "internet": {
            "max_results": 6,  # Reduced for speed
            "search_queries": 2,  # Reduced queries
            "timeout": 8,
            "verification_timeout": 3,  # Reduced verification time
        },
    },
    # Validation Settings
    "validation": {
        "parallel_processing": True,
        "batch_size": 10,
        "timeout_per_tool": 2,
    },
    # Caching Settings
    "caching": {
        "enable": True,
        "ttl": 300,  # 5 minutes
        "max_size": 1000,
    },
}

# Performance monitoring
PERFORMANCE_TARGETS = {
    "total_response_time": 3.0,  # 3 seconds max
    "pinecone_time": 1.0,  # 1 second max
    "internet_time": 2.0,  # 2 seconds max
    "validation_time": 0.5,  # 0.5 seconds max
}


def get_speed_config() -> Dict[str, Any]:
    """Get speed optimization configuration."""
    return SPEED_CONFIG


def get_performance_targets() -> Dict[str, float]:
    """Get performance targets."""
    return PERFORMANCE_TARGETS

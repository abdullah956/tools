"""Configuration for the Tool Discovery Scheduler."""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List


class DiscoverySchedulerConfig:
    """Configuration for the tool discovery scheduler."""

    def __init__(self):
        """Initialize scheduler configuration."""
        # Discovery timing
        self.discovery_interval_hours = int(os.getenv("DISCOVERY_INTERVAL_HOURS", "6"))
        self.max_tools_per_discovery = int(os.getenv("MAX_TOOLS_PER_DISCOVERY", "20"))
        self.max_queries_per_run = int(os.getenv("MAX_QUERIES_PER_RUN", "5"))

        # Discovery queries
        self.base_discovery_queries = [
            "new AI tools 2024",
            "latest artificial intelligence software",
            "emerging AI applications",
            "new machine learning tools",
            "AI productivity tools",
            "latest AI startups",
            "new AI platforms",
            "cutting-edge AI software",
            "AI development tools",
            "new AI automation tools",
            "latest AI content creation tools",
            "new AI video editing software",
            "emerging AI writing tools",
            "new AI design tools",
            "latest AI coding assistants",
            "AI tools for developers",
            "new AI marketing tools",
            "latest AI analytics software",
            "emerging AI chatbots",
            "new AI image generation tools",
        ]

        # Search parameters
        self.search_depth = os.getenv("DISCOVERY_SEARCH_DEPTH", "basic")
        self.max_results_per_query = int(os.getenv("MAX_RESULTS_PER_QUERY", "10"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))

        # Rate limiting
        self.delay_between_queries = float(os.getenv("DELAY_BETWEEN_QUERIES", "2.0"))
        self.delay_between_tools = float(os.getenv("DELAY_BETWEEN_TOOLS", "0.5"))

        # Validation settings
        self.validate_tools = os.getenv("VALIDATE_TOOLS", "true").lower() == "true"
        self.extract_pricing = os.getenv("EXTRACT_PRICING", "true").lower() == "true"
        self.enhance_descriptions = (
            os.getenv("ENHANCE_DESCRIPTIONS", "true").lower() == "true"
        )

        # Cleanup settings
        self.cleanup_old_tasks_hours = int(os.getenv("CLEANUP_OLD_TASKS_HOURS", "24"))
        self.max_task_history = int(os.getenv("MAX_TASK_HISTORY", "100"))

        # Error handling
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay_seconds = float(os.getenv("RETRY_DELAY_SECONDS", "5.0"))

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_discovery_details = (
            os.getenv("LOG_DISCOVERY_DETAILS", "true").lower() == "true"
        )

    def get_discovery_queries(self) -> List[str]:
        """Get the list of discovery queries."""
        return self.base_discovery_queries[: self.max_queries_per_run]

    def get_search_params(self) -> Dict[str, Any]:
        """Get search parameters for discovery."""
        return {
            "search_depth": self.search_depth,
            "max_results": self.max_results_per_query,
            "validate_tools": self.validate_tools,
            "extract_pricing": self.extract_pricing,
            "similarity_threshold": self.similarity_threshold,
        }

    def get_rate_limits(self) -> Dict[str, float]:
        """Get rate limiting parameters."""
        return {
            "delay_between_queries": self.delay_between_queries,
            "delay_between_tools": self.delay_between_tools,
            "retry_delay": self.retry_delay_seconds,
        }

    def get_cleanup_config(self) -> Dict[str, int]:
        """Get cleanup configuration."""
        return {
            "cleanup_old_tasks_hours": self.cleanup_old_tasks_hours,
            "max_task_history": self.max_task_history,
        }

    def should_run_discovery(self, last_run: datetime = None) -> bool:
        """Check if discovery should run based on last run time."""
        if last_run is None:
            return True

        time_since_last_run = datetime.now() - last_run
        return time_since_last_run >= timedelta(hours=self.discovery_interval_hours)

    def get_next_run_time(self, last_run: datetime = None) -> datetime:
        """Get the next scheduled run time."""
        if last_run is None:
            return datetime.now()

        return last_run + timedelta(hours=self.discovery_interval_hours)

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "discovery_interval_hours": self.discovery_interval_hours,
            "max_tools_per_discovery": self.max_tools_per_discovery,
            "max_queries_per_run": self.max_queries_per_run,
            "search_depth": self.search_depth,
            "max_results_per_query": self.max_results_per_query,
            "similarity_threshold": self.similarity_threshold,
            "delay_between_queries": self.delay_between_queries,
            "delay_between_tools": self.delay_between_tools,
            "validate_tools": self.validate_tools,
            "extract_pricing": self.extract_pricing,
            "enhance_descriptions": self.enhance_descriptions,
            "cleanup_old_tasks_hours": self.cleanup_old_tasks_hours,
            "max_task_history": self.max_task_history,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "log_level": self.log_level,
            "log_discovery_details": self.log_discovery_details,
            "discovery_queries_count": len(self.base_discovery_queries),
        }


# Global configuration instance
discovery_config = DiscoverySchedulerConfig()

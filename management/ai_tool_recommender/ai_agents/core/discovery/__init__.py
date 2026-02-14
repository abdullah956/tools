"""Tool discovery and auto-discovery services."""

from .discovery_config import DiscoverySchedulerConfig, discovery_config
from .tool_discovery_service import ToolDiscoveryService, tool_discovery_service

# Convenience access to common config values
DISCOVERY_INTERVAL_HOURS = discovery_config.discovery_interval_hours
MAX_TOOLS_PER_DISCOVERY = discovery_config.max_tools_per_discovery
DEFAULT_DISCOVERY_QUERIES = discovery_config.get_discovery_queries()

__all__ = [
    "ToolDiscoveryService",
    "tool_discovery_service",
    "DiscoverySchedulerConfig",
    "discovery_config",
    "DISCOVERY_INTERVAL_HOURS",
    "MAX_TOOLS_PER_DISCOVERY",
    "DEFAULT_DISCOVERY_QUERIES",
]

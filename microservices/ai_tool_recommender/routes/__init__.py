"""Routes for the AI tool recommender."""

# Import all route modules from views
from .views import (
    add_tools,
    discovery,
    excel_handler,
    explain,
    health_routes,
    internet_search_routes,
    search_tool,
)

__all__ = [
    "add_tools",
    "explain",
    "search_tool",
    "health_routes",
    "internet_search_routes",
    "discovery",
    "excel_handler",
]

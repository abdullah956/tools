"""Views package for organizing route modules."""

# Import routers from submodules
from .health import health_routes
from .internet import internet_search_routes
from .search import search_tool
from .tools import add_tools, discovery, excel_handler, explain

# Export all routers
__all__ = [
    "health_routes",
    "internet_search_routes",
    "search_tool",
    "add_tools",
    "explain",
    "discovery",
    "excel_handler",
]

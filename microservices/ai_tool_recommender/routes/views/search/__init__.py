"""Search-related views and routes."""

from .search_tool import router as search_tool_router

# Export router
search_tool = search_tool_router

__all__ = ["search_tool"]

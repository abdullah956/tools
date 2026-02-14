"""Internet search views and routes."""

from .internet_search_routes import router as internet_search_router

# Export router
internet_search_routes = internet_search_router

__all__ = ["internet_search_routes"]

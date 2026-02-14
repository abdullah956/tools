"""Tools-related views and routes."""

from .add_tools import router as add_tools_router
from .discovery_routes import router as discovery_router
from .excel_handler_routes import router as excel_handler_router
from .explain import router as explain_router

# Export routers
add_tools = add_tools_router
explain = explain_router
discovery = discovery_router
excel_handler = excel_handler_router

__all__ = ["add_tools", "explain", "discovery", "excel_handler"]

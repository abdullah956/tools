"""Background task scheduling and management."""

from .background_scheduler import (
    add_background_task,
    background_add_new_tools_to_pinecone,
    background_cleanup_cache,
    background_discover_new_tools,
    background_enhance_existing_tools,
    background_generate_workflow,
    background_scrape_urls,
    background_update_pinecone,
    cleanup_old_tasks,
    get_all_tasks,
    get_background_tasks,
    get_task_status,
    run_scheduled_tasks,
    run_tool_discovery_now,
)

__all__ = [
    "background_add_new_tools_to_pinecone",
    "background_discover_new_tools",
    "background_enhance_existing_tools",
    "run_scheduled_tasks",
    "run_tool_discovery_now",
    "get_all_tasks",
    "get_task_status",
    "cleanup_old_tasks",
    "background_cleanup_cache",
    "background_update_pinecone",
    "add_background_task",
    "get_background_tasks",
    "background_generate_workflow",
    "background_scrape_urls",
]

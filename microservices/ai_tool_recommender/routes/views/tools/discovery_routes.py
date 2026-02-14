"""Tool Discovery Management API endpoints."""

import logging
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from microservices.ai_tool_recommender.ai_agents.core.background import (
    get_all_tasks,
    get_task_status,
    run_tool_discovery_now,
)
from microservices.ai_tool_recommender.ai_agents.core.discovery import (
    tool_discovery_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class DiscoveryConfigRequest(BaseModel):
    """Request model for updating discovery configuration."""

    interval_hours: int = None
    max_tools: int = None
    queries: list = None


class DiscoveryResponse(BaseModel):
    """Response model for discovery operations."""

    status: str
    message: str
    data: Dict[str, Any] = None


@router.get(
    "/discovery/status", response_model=DiscoveryResponse, tags=["Tool Discovery"]
)
async def get_discovery_status():
    """Get the current status of tool discovery service."""
    try:
        status = await tool_discovery_service.get_discovery_status()

        return DiscoveryResponse(
            status="success",
            message="Discovery status retrieved successfully",
            data=status,
        )

    except Exception as e:
        logger.error(f"Error getting discovery status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/discovery/run", response_model=DiscoveryResponse, tags=["Tool Discovery"]
)
async def run_discovery_now(background_tasks: BackgroundTasks):
    """Run tool discovery immediately."""
    try:
        result = await run_tool_discovery_now()

        if result["status"] == "queued":
            return DiscoveryResponse(
                status="success",
                message="Tool discovery queued successfully",
                data=result,
            )
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Unknown error")
            )

    except Exception as e:
        logger.error(f"Error running tool discovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/discovery/config", response_model=DiscoveryResponse, tags=["Tool Discovery"]
)
async def update_discovery_config(config: DiscoveryConfigRequest):
    """Update tool discovery configuration."""
    try:
        tool_discovery_service.update_discovery_config(
            interval_hours=config.interval_hours,
            max_tools=config.max_tools,
            queries=config.queries,
        )

        return DiscoveryResponse(
            status="success",
            message="Discovery configuration updated successfully",
            data={
                "interval_hours": config.interval_hours,
                "max_tools": config.max_tools,
                "queries_count": len(config.queries) if config.queries else None,
            },
        )

    except Exception as e:
        logger.error(f"Error updating discovery config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discovery/tasks", response_model=DiscoveryResponse, tags=["Tool Discovery"]
)
async def get_discovery_tasks():
    """Get all discovery-related background tasks."""
    try:
        all_tasks = get_all_tasks()

        # Filter discovery-related tasks
        discovery_tasks = {
            task_id: task_info
            for task_id, task_info in all_tasks.items()
            if "discovery" in task_id.lower() or "enhancement" in task_id.lower()
        }

        return DiscoveryResponse(
            status="success",
            message="Discovery tasks retrieved successfully",
            data={
                "total_tasks": len(all_tasks),
                "discovery_tasks": len(discovery_tasks),
                "tasks": discovery_tasks,
            },
        )

    except Exception as e:
        logger.error(f"Error getting discovery tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discovery/tasks/{task_id}",
    response_model=DiscoveryResponse,
    tags=["Tool Discovery"],
)
async def get_discovery_task_status(task_id: str):
    """Get the status of a specific discovery task."""
    try:
        task_status = get_task_status(task_id)

        if task_status is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return DiscoveryResponse(
            status="success",
            message="Task status retrieved successfully",
            data=task_status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/discovery/test", response_model=DiscoveryResponse, tags=["Tool Discovery"]
)
async def test_discovery_service():
    """Test the discovery service with a single query."""
    try:
        # Temporarily modify discovery queries for testing
        original_queries = tool_discovery_service.discovery_queries
        tool_discovery_service.discovery_queries = ["AI tools 2024"]

        # Run discovery
        result = await tool_discovery_service.discover_new_tools()

        # Restore original queries
        tool_discovery_service.discovery_queries = original_queries

        return DiscoveryResponse(
            status="success", message="Discovery test completed", data=result
        )

    except Exception as e:
        logger.error(f"Error testing discovery service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

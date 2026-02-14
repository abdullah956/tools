"""Search tools endpoint."""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from microservices.ai_tool_recommender.ai_agents.core.background import (
    background_add_new_tools_to_pinecone,
    get_all_tasks,
    get_task_status,
)
from microservices.ai_tool_recommender.ai_agents.tools.ai_tool_recommender import (
    AIToolRecommender,
)
from microservices.ai_tool_recommender.serializers.ai_tools import SearchQuery

logger = logging.getLogger(__name__)

router = APIRouter()

# Concurrency control - limit to 10 concurrent OpenAI API calls
API_SEMAPHORE = asyncio.Semaphore(10)

# Simple in-memory cache with 5-minute TTL
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes


def _get_cache_key(query: str) -> str:
    """Generate cache key for query."""
    return hashlib.md5(query.lower().encode(), usedforsecurity=False).hexdigest()


def _is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
    """Check if cache entry is still valid."""
    return time.time() - cache_entry["timestamp"] < CACHE_TTL


async def _search_with_timeout(query: str, max_retries: int = 3) -> Dict[str, Any]:
    """Execute search with timeout protection and retry logic."""
    for attempt in range(max_retries):
        try:
            # Use asyncio.wait_for to add timeout protection
            ai_recommender = AIToolRecommender()
            return await asyncio.wait_for(
                ai_recommender.search_tools(query),
                timeout=90.0,  # Increased timeout for complex queries with refinement
            )
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                print(
                    f"Search timeout (attempt {attempt + 1}/{max_retries}), retrying..."
                )
                await asyncio.sleep(0.5)  # Wait 0.5 seconds before retry
                continue
            else:
                return {
                    "status": "error",
                    "message": "Search timeout after retries",
                    "tools": [],
                }
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Search error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying..."
                )
                await asyncio.sleep(0.5)  # Wait 0.5 seconds before retry
                continue
            else:
                return {"status": "error", "message": str(e), "tools": []}

    return {"status": "error", "message": "All retry attempts failed", "tools": []}


@router.post("/search_tools/", tags=["Tools"])
async def search_tool(
    query: SearchQuery, background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Search for AI tools based on a query with concurrency control and caching."""
    # Check cache first
    cache_key = _get_cache_key(query.query)
    if cache_key in CACHE and _is_cache_valid(CACHE[cache_key]):
        cached_result = CACHE[cache_key]["data"]
        print(f"Cache hit for query: {query.query[:50]}...")
        return {
            "status": "success",
            "workflow": cached_result.get("workflow"),
            "query": query.query,
            "cached": True,
            "message": f"Generated workflow with {len(cached_result.get('workflow', {}).get('nodes', []))} nodes from cache",
        }

    # Acquire semaphore to limit concurrent API calls
    async with API_SEMAPHORE:
        print(
            f"Processing search query: {query.query[:50]}... (Active: {10 - API_SEMAPHORE._value}/10)"
        )

        # Execute search with timeout protection using original query
        response = await _search_with_timeout(query.query)

        # Check if the response contains an error
        if response.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=response.get("message", "An error occurred during search"),
            )

        # Generate workflow with best tools and proper sequence
        workflow = None
        if response.get("tools"):
            try:
                # Import here to avoid circular imports
                from microservices.ai_tool_recommender.ai_agents.tools.ai_tool_recommender import (
                    AIToolRecommender,
                )

                # Generate workflow with reduced timeout to prevent gateway timeouts
                ai_recommender = AIToolRecommender()
                workflow = await asyncio.wait_for(
                    ai_recommender.generate_workflow(
                        query.query, response.get("tools", [])
                    ),
                    timeout=120.0,  # 2 minutes - reduced from 4 to prevent gateway timeouts
                )

                if workflow:
                    logger.info(
                        f"Generated workflow with {len(workflow.get('nodes', []))} nodes"
                    )
                else:
                    logger.warning("Workflow generation failed")

            except asyncio.TimeoutError:
                logger.warning("Workflow generation timed out, using fallback")
                workflow = None
            except Exception as e:
                logger.error(f"Error generating workflow: {e}")
                workflow = None

            # Add background task to add new internet search tools to Pinecone
            if response.get("tools"):
                # Separate internet search tools from Pinecone tools
                internet_tools = [
                    tool
                    for tool in response.get("tools", [])
                    if "Internet Search" in tool.get("Source", "")
                ]

                if internet_tools:
                    logger.info(
                        f"🌐 INTERNET TOOLS FOUND: {len(internet_tools)} tools from internet search"
                    )

                    # Log each internet tool
                    for i, tool in enumerate(internet_tools, 1):
                        title = tool.get("Title", "Unknown")
                        website = tool.get("Website", "No website")
                        logger.info(
                            f"🌐 Internet tool {i}/{len(internet_tools)}: '{title}' from {website}"
                        )

                    background_tasks.add_task(
                        background_add_new_tools_to_pinecone,
                        internet_tools,
                        query.query,
                    )
                    logger.info(
                        f"🔄 BACKGROUND TASK QUEUED: {len(internet_tools)} internet search tools will be processed for Pinecone addition"
                    )
                else:
                    logger.info("ℹ️ No internet search tools found to add to Pinecone")

        # Cache successful results with workflow
        cache_data = {
            "tools": response.get("tools", []),
            "workflow": workflow,
            "message": response.get("message", ""),
            "count": response.get("count", 0),
        }
        CACHE[cache_key] = {"data": cache_data, "timestamp": time.time()}

        # Clean old cache entries (simple cleanup)
        if len(CACHE) > 100:  # Limit cache size
            old_keys = [k for k, v in CACHE.items() if not _is_cache_valid(v)]
            for k in old_keys:
                del CACHE[k]

        # Include user information in the response with workflow
        internet_tools_count = len(
            [
                tool
                for tool in response.get("tools", [])
                if "Internet Search" in tool.get("Source", "")
            ]
        )

        return {
            "status": "success",
            "workflow": workflow,
            "query": query.query,
            "cached": False,
            "message": (
                f"Generated workflow with {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges"
                if workflow
                else "No workflow generated"
            ),
            "new_tools_discovered": internet_tools_count,
            "auto_discovery": {
                "enabled": True,
                "new_tools_queued": internet_tools_count,
                "message": (
                    f"{internet_tools_count} new tools from internet search will be added to Pinecone"
                    if internet_tools_count > 0
                    else "No new tools discovered"
                ),
            },
        }


@router.get("/background-tasks/", tags=["Background Tasks"])
async def get_background_tasks_status():
    """Get status of all background tasks."""
    try:
        tasks = get_all_tasks()
        return {"status": "success", "total_tasks": len(tasks), "tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-discovery/status", tags=["Auto Discovery"])
async def get_auto_discovery_status():
    """Get status of auto-discovery feature."""
    try:
        # Get recent background tasks related to tool addition
        all_tasks = get_all_tasks()

        # Filter for tool addition tasks
        tool_addition_tasks = {
            task_id: task_info
            for task_id, task_info in all_tasks.items()
            if "add_new_tools" in task_id.lower()
        }

        # Get discovery service status
        from microservices.ai_tool_recommender.ai_agents.core.discovery import (
            tool_discovery_service,
        )

        discovery_status = await tool_discovery_service.get_discovery_status()

        return {
            "status": "success",
            "auto_discovery": {
                "enabled": True,
                "description": "New tools from internet search are automatically added to Pinecone",
                "recent_tasks": len(tool_addition_tasks),
                "discovery_service_status": discovery_status,
            },
            "recent_tool_addition_tasks": tool_addition_tasks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/background-tasks/{task_id}", tags=["Background Tasks"])
async def get_background_task_status(task_id: str):
    """Get status of a specific background task."""
    try:
        task_status = get_task_status(task_id)
        if task_status is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return {"status": "success", "task": task_status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

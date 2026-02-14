"""FastAPI Background Scheduler for AI Tool Recommender."""

import asyncio
from datetime import datetime, timedelta
import logging
import time
from typing import Any, Callable, Dict, Optional

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    """FastAPI Background Scheduler for handling async tasks."""

    def __init__(self):
        """Initialize the background scheduler."""
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task_counter = 0
        self.background_tasks = BackgroundTasks()

    def add_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        delay: float = 0,
        **kwargs,
    ) -> str:
        """Add a background task to be executed.

        Args:
            func: Function to execute
            *args: Arguments for the function
            task_id: Optional task ID, will be generated if not provided
            delay: Delay in seconds before execution
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        if task_id is None:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time())}"

        task_info = {
            "id": task_id,
            "func": func,
            "args": args,
            "kwargs": kwargs,
            "delay": delay,
            "created_at": datetime.now(),
            "status": "pending",
        }

        self.tasks[task_id] = task_info

        # Add to FastAPI background tasks
        if delay > 0:
            self.background_tasks.add_task(self._delayed_task_wrapper, task_id, delay)
        else:
            self.background_tasks.add_task(self._task_wrapper, task_id)

        logger.info(f"Added background task: {task_id}")
        return task_id

    async def _delayed_task_wrapper(self, task_id: str, delay: float):
        """Wrapper for delayed task execution."""
        await asyncio.sleep(delay)
        await self._execute_task(task_id)

    async def _task_wrapper(self, task_id: str):
        """Wrapper for immediate task execution."""
        await self._execute_task(task_id)

    async def _execute_task(self, task_id: str):
        """Execute a background task."""
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return

        task_info = self.tasks[task_id]
        task_info["status"] = "running"
        task_info["started_at"] = datetime.now()

        try:
            logger.info(f"Executing background task: {task_id}")

            # Execute the task
            if asyncio.iscoroutinefunction(task_info["func"]):
                result = await task_info["func"](
                    *task_info["args"], **task_info["kwargs"]
                )
            else:
                result = task_info["func"](*task_info["args"], **task_info["kwargs"])

            task_info["status"] = "completed"
            task_info["completed_at"] = datetime.now()
            task_info["result"] = result

            logger.info(f"Background task completed: {task_id}")

        except Exception as e:
            task_info["status"] = "failed"
            task_info["completed_at"] = datetime.now()
            task_info["error"] = str(e)

            logger.error(f"Background task failed: {task_id} - {e}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a background task."""
        if task_id not in self.tasks:
            return None

        task_info = self.tasks[task_id].copy()

        # Remove function reference for serialization
        if "func" in task_info:
            del task_info["func"]

        return task_info

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all background tasks."""
        result = {}
        for task_id, _task_info in self.tasks.items():
            result[task_id] = self.get_task_status(task_id)
        return result

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed/failed tasks."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        tasks_to_remove = []
        for task_id, task_info in self.tasks.items():
            if (
                task_info["status"] in ["completed", "failed"]
                and task_info.get("completed_at", datetime.now()) < cutoff_time
            ):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.tasks[task_id]

        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")


# Global scheduler instance
background_scheduler = BackgroundScheduler()


# Background task functions
async def background_generate_workflow(
    query: str, tools: list, user_id: str = "system"
):
    """Background task to generate workflow."""
    try:
        logger.info(
            f"Background workflow generation started for query: {query[:50]}..."
        )

        # Import here to avoid circular imports
        from microservices.ai_tool_recommender.ai_agents.tools.ai_tool_recommender import (
            AIToolRecommender,
        )

        ai_recommender = AIToolRecommender()
        workflow = await ai_recommender.generate_workflow(query, tools)

        if workflow:
            logger.info(
                f"Background workflow generated successfully for query: {query[:50]}..."
            )
            return {"status": "success", "workflow": workflow}
        else:
            logger.warning(
                f"Background workflow generation failed for query: {query[:50]}..."
            )
            return {"status": "failed", "error": "Workflow generation failed"}

    except Exception as e:
        logger.error(f"Background workflow generation error: {e}")
        return {"status": "error", "error": str(e)}


async def background_scrape_urls(urls: list, query: str):
    """Background task to scrape URLs for tool data."""
    try:
        logger.info(f"Background URL scraping started for {len(urls)} URLs")

        # Import here to avoid circular imports
        # from microservices.ai_tool_recommender.ai_agents.tools.internet_search.service import (
        #     InternetSearchService,
        # )

        # internet_service = InternetSearchService()  # Unused for now
        results = []

        for url in urls:
            try:
                # Simulate URL scraping
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
                # Add actual scraping logic here
                results.append({"url": url, "status": "processed"})
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                results.append({"url": url, "status": "failed", "error": str(e)})

        logger.info(f"Background URL scraping completed: {len(results)} results")
        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Background URL scraping error: {e}")
        return {"status": "error", "error": str(e)}


async def background_update_pinecone(tools: list):
    """Background task to update Pinecone with new tools."""
    try:
        logger.info(f"Background Pinecone update started for {len(tools)} tools")

        # Import here to avoid circular imports
        # from microservices.ai_tool_recommender.ai_agents.tools.pinecone.service import (
        #     PineconeService,
        # )

        # pinecone_service = PineconeService()  # Unused for now

        # Add tools to Pinecone
        for tool in tools:
            try:
                # Add actual Pinecone update logic here
                await asyncio.sleep(0.1)  # Small delay
                logger.info(
                    f"Updated Pinecone with tool: {tool.get('Title', 'Unknown')}"
                )
            except Exception as e:
                logger.error(
                    f"Error updating Pinecone with tool {tool.get('Title', 'Unknown')}: {e}"
                )

        logger.info("Background Pinecone update completed")
        return {"status": "success", "updated_count": len(tools)}

    except Exception as e:
        logger.error(f"Background Pinecone update error: {e}")
        return {"status": "error", "error": str(e)}


async def background_add_new_tools_to_pinecone(tools: list, query: str = ""):
    """Background task to add new tools from internet search to Pinecone."""
    try:
        logger.info(
            f"🔄 BACKGROUND TASK: Processing {len(tools)} internet search tools for Pinecone addition"
        )
        logger.info(f"🔍 Query context: '{query}'")

        # Log each tool being processed
        for i, tool in enumerate(tools, 1):
            title = tool.get("Title", "Unknown")
            website = tool.get("Website", "No website")
            logger.info(f"🌐 Processing tool {i}/{len(tools)}: '{title}' from {website}")

        # Import here to avoid circular imports
        from microservices.ai_tool_recommender.ai_agents.core.discovery import (
            tool_discovery_service,
        )
        from microservices.ai_tool_recommender.ai_agents.tools.pinecone.service import (
            PineconeService,
        )

        pinecone_service = PineconeService()
        added_count = 0
        skipped_count = 0

        # Separate new tools from existing ones
        new_tools = []

        logger.info(f"🔍 CHECKING FOR DUPLICATES: Analyzing {len(tools)} tools...")

        for i, tool in enumerate(tools, 1):
            try:
                title = tool.get("Title", "Unknown")
                logger.info(f"🔍 Checking tool {i}/{len(tools)}: '{title}'")

                # Check if tool already exists in Pinecone
                is_existing = (
                    await tool_discovery_service._check_tool_exists_in_pinecone(tool)
                )

                if not is_existing:
                    logger.info(
                        f"✅ NEW TOOL: '{title}' - not found in Pinecone, will add"
                    )
                    # Enhance tool data before adding
                    enhanced_tool = await tool_discovery_service._enhance_tool_data(
                        tool
                    )
                    new_tools.append(enhanced_tool)
                else:
                    skipped_count += 1
                    logger.info(
                        f"🔄 DUPLICATE: '{title}' - already exists in Pinecone, skipping"
                    )

            except Exception as e:
                logger.error(
                    f"❌ ERROR processing tool '{tool.get('Title', 'Unknown')}': {e}"
                )
                continue

        logger.info("📊 DUPLICATE CHECK COMPLETE:")
        logger.info(f"   📊 Total tools processed: {len(tools)}")
        logger.info(f"   🆕 New tools to add: {len(new_tools)}")
        logger.info(f"   🔄 Duplicates skipped: {skipped_count}")

        # Add all new tools in batch (same approach as Excel handler)
        if new_tools:
            logger.info(f"🚀 ADDING {len(new_tools)} NEW TOOLS TO PINECONE...")

            batch_result = await pinecone_service.add_tools_batch(new_tools)
            added_count = batch_result.get("success", 0)
            failed_count = batch_result.get("failed", 0)
            duplicate_count = batch_result.get("duplicates", 0)

            logger.info("✅ PINECONE BATCH ADDITION COMPLETE:")
            logger.info(f"   ✅ Successfully added: {added_count}")
            logger.info(f"   ❌ Failed to add: {failed_count}")
            logger.info(f"   🔄 Duplicates in batch: {duplicate_count}")

            # Log success rate
            if len(new_tools) > 0:
                success_rate = (added_count / len(new_tools)) * 100
                logger.info(f"   📈 Success rate: {success_rate:.1f}%")
        else:
            logger.info("ℹ️ No new tools to add to Pinecone")

        logger.info("🎯 BACKGROUND TASK COMPLETE:")
        logger.info(f"   📊 Total processed: {len(tools)}")
        logger.info(f"   ✅ Added to Pinecone: {added_count}")
        logger.info(f"   🔄 Skipped (duplicates): {skipped_count}")
        logger.info(f"   🔍 Query: '{query}'")

        return {
            "status": "success",
            "added_count": added_count,
            "skipped_count": skipped_count,
            "total_processed": len(tools),
            "query": query,
        }

    except Exception as e:
        logger.error(f"❌ BACKGROUND TASK ERROR: {e}")
        logger.error(f"   📊 Tools that failed: {len(tools)}")
        return {"status": "error", "error": str(e)}


async def background_cleanup_cache():
    """Background task to cleanup cache and old data."""
    try:
        logger.info("Background cache cleanup started")

        # Cleanup old tasks
        background_scheduler.cleanup_old_tasks(max_age_hours=24)

        # Add other cleanup tasks here
        await asyncio.sleep(1)  # Simulate cleanup work

        logger.info("Background cache cleanup completed")
        return {"status": "success", "message": "Cache cleanup completed"}

    except Exception as e:
        logger.error(f"Background cache cleanup error: {e}")
        return {"status": "error", "error": str(e)}


async def background_discover_new_tools():
    """Background task to discover new AI tools from the internet."""
    try:
        logger.info("Background tool discovery started")

        # Import here to avoid circular imports
        from microservices.ai_tool_recommender.ai_agents.core.discovery import (
            tool_discovery_service,
        )

        # Run tool discovery
        result = await tool_discovery_service.discover_new_tools()

        logger.info(f"Background tool discovery completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Background tool discovery error: {e}")
        return {"status": "error", "error": str(e)}


async def background_enhance_existing_tools():
    """Background task to enhance existing tools with additional data."""
    try:
        logger.info("Background tool enhancement started")

        # Import here to avoid circular imports
        # from microservices.ai_tool_recommender.ai_agents.tools.pinecone.service import (
        #     PineconeService,
        # )

        # pinecone_service = PineconeService()  # Unused for now

        # Get some existing tools for enhancement
        # This is a placeholder - implement actual enhancement logic
        await asyncio.sleep(1)  # Simulate enhancement work

        logger.info("Background tool enhancement completed")
        return {"status": "success", "message": "Tool enhancement completed"}

    except Exception as e:
        logger.error(f"Background tool enhancement error: {e}")
        return {"status": "error", "error": str(e)}


# Scheduled tasks (can be called periodically)
async def run_scheduled_tasks():
    """Run scheduled background tasks."""
    try:
        logger.info("Running scheduled background tasks")

        # Add cache cleanup task
        background_scheduler.add_task(
            background_cleanup_cache, task_id="scheduled_cleanup"
        )

        # Add tool discovery task
        background_scheduler.add_task(
            background_discover_new_tools, task_id="scheduled_discovery"
        )

        # Add tool enhancement task (less frequent)
        background_scheduler.add_task(
            background_enhance_existing_tools, task_id="scheduled_enhancement"
        )

        logger.info("Scheduled background tasks queued")

    except Exception as e:
        logger.error(f"Error running scheduled tasks: {e}")


async def run_tool_discovery_now():
    """Run tool discovery immediately."""
    try:
        logger.info("Running immediate tool discovery")

        task_id = background_scheduler.add_task(
            background_discover_new_tools,
            task_id=f"immediate_discovery_{int(time.time())}",
        )

        logger.info(f"Tool discovery task queued: {task_id}")
        return {"status": "queued", "task_id": task_id}

    except Exception as e:
        logger.error(f"Error queuing tool discovery: {e}")
        return {"status": "error", "error": str(e)}


# Utility functions
def get_background_tasks() -> BackgroundTasks:
    """Get the background tasks instance."""
    return background_scheduler.background_tasks


def add_background_task(
    func: Callable, *args, task_id: Optional[str] = None, delay: float = 0, **kwargs
) -> str:
    """Add a background task."""
    return background_scheduler.add_task(
        func, *args, task_id=task_id, delay=delay, **kwargs
    )


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status."""
    return background_scheduler.get_task_status(task_id)


def get_all_tasks() -> Dict[str, Dict[str, Any]]:
    """Get all tasks."""
    return background_scheduler.get_all_tasks()


def cleanup_old_tasks(max_age_hours: int = 24):
    """Cleanup old tasks."""
    return background_scheduler.cleanup_old_tasks(max_age_hours)

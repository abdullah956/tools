"""Helper service to submit internet-discovered tools to the scraping pipeline."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class InternetToolSubmitter:
    """Service to submit internet-discovered tools for background scraping."""

    def __init__(self):
        """Initialize the submitter."""
        pass

    def submit_tools(
        self, tools: List[Dict[str, Any]], source_query: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Submit tools to the scraping pipeline using Django internal methods.

        Args:
            tools: List of tool dictionaries with Title, Description, Website, etc.
            source_query: The original search query that discovered these tools

        Returns:
            Response data with job info or None if failed
        """
        if not tools:
            logger.warning("No tools to submit")
            return None

        try:
            # Import Django models and tasks here to avoid circular imports
            from django.utils import timezone

            from management.tool_scraping.models import ScrapingJob
            from management.tool_scraping.tasks import process_internet_discovered_tools

            logger.info(f"🚀 Submitting {len(tools)} tools to scraping pipeline")

            # Transform tools to match expected format (lowercase keys)
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "website": tool.get("Website", ""),
                    "title": tool.get("Title", ""),
                    "description": tool.get("Description", ""),
                    "category": tool.get("Category", ""),
                    "features": tool.get("Features", ""),
                    "source": tool.get("Source", "Internet Search"),
                    "relevance_score": tool.get("Relevance Score", 0),
                    "twitter": tool.get("Twitter", ""),
                    "facebook": tool.get("Facebook", ""),
                    "linkedin": tool.get("Linkedin", ""),
                    "instagram": tool.get("Instagram", ""),
                }
                formatted_tools.append(formatted_tool)

            # Create a scraping job with tools data in payload
            job = ScrapingJob.objects.create(
                job_type=ScrapingJob.JobType.INTERNET_DISCOVERY,
                status=ScrapingJob.Status.PENDING,
                payload={
                    "source_query": source_query,
                    "tools": formatted_tools,
                    "tool_count": len(formatted_tools),
                },
                logs=[
                    f"Internet discovery job created for query: {source_query}",
                    f"Found {len(formatted_tools)} tools to process.",
                ],
                started_at=timezone.now(),
            )

            logger.info(
                f"✅ Created scraping job {job.id} for {len(formatted_tools)} tools"
            )

            # Trigger Celery task to process the tools
            process_internet_discovered_tools.delay(str(job.id))

            logger.info(
                f"✅ Successfully queued {len(formatted_tools)} tools for scraping. "
                f"Job ID: {job.id}"
            )

            return {
                "id": str(job.id),
                "job_type": job.job_type,
                "status": job.status,
                "tool_count": len(formatted_tools),
                "source_query": source_query,
            }

        except Exception as e:
            logger.error(f"❌ Error submitting tools to scraping pipeline: {e}")
            return None

    async def submit_tools_async(
        self, tools: List[Dict[str, Any]], source_query: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Async version of submit_tools.

        Args:
            tools: List of tool dictionaries
            source_query: The original search query

        Returns:
            Response data or None if failed
        """
        import asyncio

        # Run the sync version in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.submit_tools, tools, source_query)


# Singleton instance for easy access
internet_tool_submitter = InternetToolSubmitter()

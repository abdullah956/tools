"""Tool Discovery Service for automatically finding and adding new AI tools."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolDiscoveryService:
    """Service for discovering new AI tools from the internet and adding them to Pinecone."""

    def __init__(self):
        """Initialize the tool discovery service."""
        self.discovery_queries = [
            "new AI tools 2024",
            "latest artificial intelligence software",
            "emerging AI applications",
            "new machine learning tools",
            "AI productivity tools",
            "latest AI startups",
            "new AI platforms",
            "cutting-edge AI software",
            "AI development tools",
            "new AI automation tools",
            "latest AI content creation tools",
            "new AI video editing software",
            "emerging AI writing tools",
            "new AI design tools",
            "latest AI coding assistants",
        ]
        self.last_discovery_run = None
        self.discovery_interval_hours = 6  # Run every 6 hours
        self.max_tools_per_discovery = 20
        self._pinecone_service = None
        self._internet_service = None
        self._llm = None

    @property
    def pinecone_service(self):
        """Lazy load Pinecone service."""
        if self._pinecone_service is None:
            from ai_tool_recommender.ai_agents.tools.pinecone.service import (
                PineconeService,
            )

            self._pinecone_service = PineconeService()
        return self._pinecone_service

    @property
    def internet_service(self):
        """Lazy load Internet Search service."""
        if self._internet_service is None:
            from ai_tool_recommender.ai_agents.tools.internet_search.service import (
                InternetSearchService,
            )

            self._internet_service = InternetSearchService()
        return self._internet_service

    @property
    def llm(self):
        """Lazy load LLM service."""
        if self._llm is None:
            from ai_tool_recommender.ai_agents.core.llm import shared_llm

            self._llm = shared_llm
        return self._llm

    async def discover_new_tools(self) -> Dict[str, Any]:
        """Discover new AI tools from the internet and add them to Pinecone.

        Returns:
            Dictionary with discovery results
        """
        try:
            logger.info("Starting tool discovery process")
            start_time = time.time()

            # Check if we should run discovery
            if not self._should_run_discovery():
                logger.info("Tool discovery skipped - too soon since last run")
                return {
                    "status": "skipped",
                    "message": "Discovery skipped - too soon since last run",
                    "last_run": self.last_discovery_run,
                }

            discovered_tools = []
            total_processed = 0

            # Process discovery queries
            for query in self.discovery_queries[:5]:  # Limit to 5 queries per run
                try:
                    logger.info(f"Discovering tools with query: {query}")

                    # Search for tools using internet search
                    search_results = await self.internet_service.search_ai_tools(
                        query=query, max_results=10
                    )

                    if search_results:
                        tools = search_results
                        total_processed += len(tools)

                        # Check which tools are new (not in Pinecone)
                        new_tools = await self._filter_new_tools(tools)

                        if new_tools:
                            logger.info(
                                f"Found {len(new_tools)} new tools from query: {query}"
                            )
                            discovered_tools.extend(new_tools)

                        # Small delay between queries to avoid rate limiting
                        await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Error processing discovery query '{query}': {e}")
                    continue

            # Limit the number of tools to process
            if len(discovered_tools) > self.max_tools_per_discovery:
                discovered_tools = discovered_tools[: self.max_tools_per_discovery]
                logger.info(
                    f"Limited discovery to {self.max_tools_per_discovery} tools"
                )

            # Add new tools to Pinecone
            added_count = 0
            if discovered_tools:
                added_count = await self._add_tools_to_pinecone(discovered_tools)

            # Update last discovery run time
            self.last_discovery_run = datetime.now()

            execution_time = time.time() - start_time

            result = {
                "status": "success",
                "discovered_tools": len(discovered_tools),
                "added_to_pinecone": added_count,
                "total_processed": total_processed,
                "execution_time_seconds": round(execution_time, 2),
                "last_discovery_run": self.last_discovery_run.isoformat(),
                "tools_added": [
                    {
                        "title": tool.get("Title", "Unknown"),
                        "website": tool.get("Website", ""),
                        "source": tool.get("Source", "Internet Search"),
                    }
                    for tool in discovered_tools[:10]  # Show first 10
                ],
            }

            logger.info(f"Tool discovery completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Tool discovery error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_discovery_run": (
                    self.last_discovery_run.isoformat()
                    if self.last_discovery_run
                    else None
                ),
            }

    def _should_run_discovery(self) -> bool:
        """Check if discovery should run based on last run time."""
        if self.last_discovery_run is None:
            return True

        time_since_last_run = datetime.now() - self.last_discovery_run
        return time_since_last_run >= timedelta(hours=self.discovery_interval_hours)

    async def _filter_new_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter tools to find only new ones not already in Pinecone.

        Args:
            tools: List of tools from internet search

        Returns:
            List of new tools not in Pinecone
        """
        try:
            new_tools = []

            for tool in tools:
                # Check if tool already exists in Pinecone
                is_existing = await self._check_tool_exists_in_pinecone(tool)

                if not is_existing:
                    # Enhance tool data before adding
                    enhanced_tool = await self._enhance_tool_data(tool)
                    new_tools.append(enhanced_tool)
                else:
                    logger.debug(
                        f"Tool already exists in Pinecone: {tool.get('Title', 'Unknown')}"
                    )

            return new_tools

        except Exception as e:
            logger.error(f"Error filtering new tools: {e}")
            return tools  # Return all tools if filtering fails

    async def _check_tool_exists_in_pinecone(self, tool: Dict[str, Any]) -> bool:
        """Check if a tool already exists in Pinecone with enhanced deduplication logic.

        Args:
            tool: Tool data to check

        Returns:
            True if tool exists, False otherwise
        """
        try:
            # Use multiple identifiers for better deduplication
            title = tool.get("Title", "").strip().lower()
            website = tool.get("Website", "").strip().lower()
            description = tool.get("Description", "").strip().lower()

            if not title and not website:
                logger.warning(
                    "Tool has no title or website, skipping deduplication check"
                )
                return False

            # Create multiple search queries for comprehensive checking
            search_queries = []

            if title:
                search_queries.append(title)

            # Only use website as search query if it's a clean domain (not a redirect URL)
            if website and not (
                "redirect" in website
                or "grounding-api" in website
                or len(website) > 100
            ):
                search_queries.append(website)
                if title:
                    search_queries.append(f"{title} {website}")

            # Add partial matches for better detection
            if title and len(title) > 10:
                # Check for partial title matches
                words = title.split()
                if len(words) > 1:
                    search_queries.extend(words[:2])  # First two words

            # Search Pinecone for similar tools using multiple queries
            for search_query in search_queries[
                :3
            ]:  # Limit to 3 queries to avoid too many API calls
                try:
                    search_results = await self.pinecone_service.search_tools(
                        query=search_query,
                        max_results=10,  # Get more results for better matching
                        similarity_threshold=0.045,  # Proper threshold for Pinecone
                    )

                    if search_results:
                        existing_tools = search_results

                        # Check for various types of matches
                        for existing_tool in existing_tools:
                            existing_title = (
                                existing_tool.get("Title", "").strip().lower()
                            )
                            existing_website = (
                                existing_tool.get("Website", "").strip().lower()
                            )
                            existing_description = (
                                existing_tool.get("Description", "").strip().lower()
                            )

                            # 1. Exact title match
                            if title and existing_title and title == existing_title:
                                logger.info(f"Exact title match found: {title}")
                                return True

                            # 2. Exact website match
                            if (
                                website
                                and existing_website
                                and website == existing_website
                            ):
                                logger.info(f"Exact website match found: {website}")
                                return True

                            # 3. Title contains or is contained in existing title
                            if (
                                title
                                and existing_title
                                and (title in existing_title or existing_title in title)
                                and len(title) > 5
                            ):
                                logger.info(
                                    f"Title similarity match found: {title} vs {existing_title}"
                                )
                                return True

                            # 4. Website domain match (extract domain from URLs)
                            if website and existing_website:
                                try:
                                    from urllib.parse import urlparse

                                    current_domain = urlparse(website).netloc.lower()
                                    existing_domain = urlparse(
                                        existing_website
                                    ).netloc.lower()

                                    if (
                                        current_domain
                                        and existing_domain
                                        and current_domain == existing_domain
                                    ):
                                        logger.info(
                                            f"Domain match found: {current_domain}"
                                        )
                                        return True
                                except Exception:
                                    pass

                            # 5. Description similarity (for tools with similar descriptions)
                            if (
                                description
                                and existing_description
                                and len(description) > 50
                            ):
                                # Check if descriptions are very similar
                                words_current = set(description.split())
                                words_existing = set(existing_description.split())
                                common_words = words_current.intersection(
                                    words_existing
                                )

                                if len(common_words) > 10:  # More than 10 common words
                                    logger.info("Description similarity match found")
                                    return True

                except Exception as e:
                    logger.error(f"Error in search query '{search_query}': {e}")
                    continue

            return False

        except Exception as e:
            logger.error(f"Error checking tool existence in Pinecone: {e}")
            # Return False to be safe - assume tool is new if check fails
            return False

    async def _enhance_tool_data(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance tool data with additional information.

        Args:
            tool: Basic tool data

        Returns:
            Enhanced tool data
        """
        try:
            # Use LLM to enhance tool description and features
            enhancement_prompt = f"""
            Enhance this AI tool data with more detailed information:

            Title: {tool.get('Title', 'Unknown')}
            Description: {tool.get('Description', '')}
            Website: {tool.get('Website', '')}
            Category: {tool.get('Category', '')}

            Please provide:
            1. A more detailed description (2-3 sentences)
            2. Key features list (5-7 features)
            3. Relevant tags/keywords (5-8 tags)
            4. Primary use case
            5. Target audience

            IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, no code blocks. Start with {{ and end with }}.

            Return as JSON:
            {{
                "enhanced_description": "",
                "key_features": [],
                "tags": [],
                "use_case": "",
                "target_audience": ""
            }}
            """

            try:
                response = await self.llm.generate_response(enhancement_prompt)

                # Clean and parse JSON response
                try:
                    # Try to parse as JSON first
                    enhancement_data = json.loads(response)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON from response
                    import re

                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    if json_match:
                        enhancement_data = json.loads(json_match.group())
                    else:
                        # Fallback: create basic enhancement data
                        enhancement_data = {
                            "enhanced_description": tool.get("Description", ""),
                            "key_features": [],
                            "tags": [],
                            "use_case": "",
                            "target_audience": "",
                        }
                        logger.warning(
                            "Could not parse LLM response as JSON, using fallback data"
                        )

                # Update tool with enhanced data
                enhanced_tool = tool.copy()
                enhanced_tool["Description"] = enhancement_data.get(
                    "enhanced_description", tool.get("Description", "")
                )
                enhanced_tool["Features"] = ", ".join(
                    enhancement_data.get("key_features", [])
                )
                enhanced_tool["Tags (Keywords)"] = ", ".join(
                    enhancement_data.get("tags", [])
                )
                enhanced_tool["Use Case"] = enhancement_data.get("use_case", "")
                enhanced_tool["Target Audience"] = enhancement_data.get(
                    "target_audience", ""
                )
                enhanced_tool["Discovery Date"] = datetime.now().isoformat()
                enhanced_tool["Source"] = "Auto-Discovery"

                return enhanced_tool

            except Exception as e:
                logger.error(f"Error enhancing tool data with LLM: {e}")
                # Return original tool with basic enhancements
                enhanced_tool = tool.copy()
                enhanced_tool["Discovery Date"] = datetime.now().isoformat()
                enhanced_tool["Source"] = "Auto-Discovery"
                return enhanced_tool

        except Exception as e:
            logger.error(f"Error enhancing tool data: {e}")
            return tool

    async def _add_tools_to_pinecone(self, tools: List[Dict[str, Any]]) -> int:
        """Add new tools to Pinecone vector database using Excel handler approach.

        Args:
            tools: List of tools to add

        Returns:
            Number of tools successfully added
        """
        try:
            logger.info(
                f"🌐 INTERNET TOOLS → PINECONE: Starting to add {len(tools)} tools from internet search"
            )

            # Log each tool before adding
            for i, tool in enumerate(tools, 1):
                title = tool.get("Title", "Unknown")
                website = tool.get("Website", "No website")
                source = tool.get("Source", "Unknown")
                logger.info(
                    f"🌐 Tool {i}/{len(tools)}: '{title}' from {website} (Source: {source})"
                )

            # Use batch method for better performance (same as Excel handler)
            batch_result = await self.pinecone_service.add_tools_batch(tools)

            # Extract detailed results
            added_count = batch_result.get("success", 0)
            failed_count = batch_result.get("failed", 0)
            duplicate_count = batch_result.get("duplicates", 0)

            # Detailed logging of results
            logger.info("✅ PINECONE ADDITION COMPLETE:")
            logger.info(f"   📊 Total processed: {len(tools)}")
            logger.info(f"   ✅ Successfully added: {added_count}")
            logger.info(f"   🔄 Duplicates skipped: {duplicate_count}")
            logger.info(f"   ❌ Failed to add: {failed_count}")

            # Log success rate
            if len(tools) > 0:
                success_rate = (added_count / len(tools)) * 100
                logger.info(f"   📈 Success rate: {success_rate:.1f}%")

            # Log individual tool results if we have detailed results
            if "details" in batch_result:
                logger.info("📋 DETAILED RESULTS:")
                for detail in batch_result["details"]:
                    status_emoji = {
                        "added": "✅",
                        "duplicate": "🔄",
                        "failed": "❌",
                        "skipped": "⏭️",
                    }.get(detail["status"], "❓")

                    logger.info(
                        f"   {status_emoji} {detail['title']}: {detail['reason']}"
                    )

            return added_count

        except Exception as e:
            logger.error(f"❌ ERROR adding internet tools to Pinecone: {e}")
            logger.error(f"   📊 Tools that failed: {len(tools)}")
            return 0

    async def get_discovery_status(self) -> Dict[str, Any]:
        """Get the current status of tool discovery."""
        return {
            "last_discovery_run": (
                self.last_discovery_run.isoformat() if self.last_discovery_run else None
            ),
            "discovery_interval_hours": self.discovery_interval_hours,
            "max_tools_per_discovery": self.max_tools_per_discovery,
            "should_run_discovery": self._should_run_discovery(),
            "discovery_queries_count": len(self.discovery_queries),
        }

    def update_discovery_config(
        self,
        interval_hours: Optional[int] = None,
        max_tools: Optional[int] = None,
        queries: Optional[List[str]] = None,
    ):
        """Update discovery configuration."""
        if interval_hours is not None:
            self.discovery_interval_hours = interval_hours
        if max_tools is not None:
            self.max_tools_per_discovery = max_tools
        if queries is not None:
            self.discovery_queries = queries

        logger.info(
            f"Updated discovery config: interval={self.discovery_interval_hours}h, max_tools={self.max_tools_per_discovery}"
        )


# Global tool discovery service instance
tool_discovery_service = ToolDiscoveryService()

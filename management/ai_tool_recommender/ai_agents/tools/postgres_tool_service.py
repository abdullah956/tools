"""PostgreSQL Tool Service - Fetches detailed tool descriptions for LLM selection."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)


class PostgresToolService:
    """Service for fetching detailed tool data from PostgreSQL database."""

    def __init__(self):
        """Initialize the PostgreSQL tool service."""
        pass

    async def enrich_tools_with_postgres_data(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich Pinecone tool results with detailed PostgreSQL data.

        Args:
            tools: List of tools from Pinecone search

        Returns:
            List of tools enriched with PostgreSQL data
        """
        try:
            if not tools:
                return tools

            logger.info(f"🔍 Enriching {len(tools)} tools with PostgreSQL data")

            # Extract tool identifiers for lookup
            tool_identifiers = []
            for tool in tools:
                # Try multiple identification methods
                tool_id = tool.get("ID") or tool.get("id")
                title = tool.get("Title") or tool.get("title", "")
                website = tool.get("Website") or tool.get("website", "")

                tool_identifiers.append(
                    {
                        "original_tool": tool,
                        "id": tool_id,
                        "title": title.strip() if title else "",
                        "website": website.strip() if website else "",
                    }
                )

            # Fetch detailed data from PostgreSQL in parallel
            # Create tasks for all tools to fetch in parallel
            fetch_tasks = [
                self._fetch_tool_from_postgres(identifier)
                for identifier in tool_identifiers
            ]

            # Execute all fetches in parallel
            postgres_results = await asyncio.gather(
                *fetch_tasks, return_exceptions=True
            )

            # Merge results with original tools
            enriched_tools = []
            for i, (identifier, postgres_data) in enumerate(
                zip(tool_identifiers, postgres_results)
            ):
                if isinstance(postgres_data, Exception):
                    # If fetch failed, keep original tool
                    logger.warning(f"Error fetching tool {i+1}: {postgres_data}")
                    enriched_tools.append(identifier["original_tool"])
                elif postgres_data:
                    # Merge Pinecone data with PostgreSQL data
                    enriched_tool = self._merge_tool_data(
                        identifier["original_tool"], postgres_data
                    )
                    enriched_tools.append(enriched_tool)
                else:
                    # Keep original tool if no PostgreSQL match
                    enriched_tools.append(identifier["original_tool"])

            enriched_count = len(
                [t for t in enriched_tools if t.get("_postgres_enriched")]
            )
            logger.info(f"✅ Enriched {enriched_count} tools with PostgreSQL data")
            return enriched_tools

        except Exception as e:
            logger.error(f"❌ Error enriching tools with PostgreSQL data: {e}")
            return tools  # Return original tools on error

    async def _fetch_tool_from_postgres(
        self, identifier: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch tool data from PostgreSQL using various identifiers.

        Args:
            identifier: Dictionary with tool identification data

        Returns:
            Tool data from PostgreSQL or None if not found
        """
        try:
            from management.tools.models import Tool

            tool_id = identifier.get("id")
            title = identifier.get("title", "")
            website = identifier.get("website", "")

            # Try to find tool using multiple methods
            tool_obj = None

            # Method 1: Direct ID lookup (most reliable)
            # Use filter().first() instead of get() to handle duplicates gracefully
            if tool_id:
                try:
                    tool_obj = await sync_to_async(
                        lambda: Tool.objects.filter(id=tool_id).first()
                    )()
                    if tool_obj:
                        logger.debug(f"✅ Found tool by ID: {tool_id}")
                except Exception as e:
                    logger.debug(f"Error fetching tool by ID {tool_id}: {e}")
                    pass

            # Method 2: Exact title match
            # Use filter().first() instead of get() to handle duplicates gracefully
            if not tool_obj and title:
                try:
                    tool_obj = await sync_to_async(
                        lambda: Tool.objects.filter(title__iexact=title.strip()).first()
                    )()
                    if tool_obj:
                        logger.debug(f"✅ Found tool by title: {title}")
                except Exception as e:
                    logger.debug(f"Error fetching tool by title {title}: {e}")
                    pass

            # Method 3: Website match
            if not tool_obj and website:
                try:
                    # Clean website URL for comparison
                    clean_website = self._clean_website_url(website)
                    if clean_website:
                        tool_obj = await sync_to_async(Tool.objects.filter)(
                            website__icontains=clean_website
                        )
                        tool_obj = await sync_to_async(lambda: tool_obj.first())()
                        if tool_obj:
                            logger.debug(f"✅ Found tool by website: {website}")
                except Exception:
                    pass

            # Method 4: Fuzzy search using PostgreSQL search capabilities
            if not tool_obj and title and len(title) > 3:
                try:
                    search_results = await sync_to_async(
                        lambda: list(Tool.search(title)[:1])
                    )()
                    if search_results and len(search_results) > 0:
                        # Check if similarity is high enough
                        result = search_results[0]
                        similarity = getattr(result, "total_similarity", 0)
                        if similarity >= 0.6:  # High similarity threshold
                            tool_obj = result
                            logger.debug(
                                f"✅ Found tool by fuzzy search: {title} "
                                f"(similarity: {similarity})"
                            )
                except Exception:
                    pass

            if not tool_obj:
                logger.debug(f"❌ No PostgreSQL match found for: {title}")
                return None

            # Convert to dictionary with detailed information
            return {
                "id": str(tool_obj.id),
                "title": tool_obj.title,
                "description": tool_obj.description,
                "category": tool_obj.category,
                "features": tool_obj.features,
                "tags": tool_obj.tags,
                "website": tool_obj.website,
                "twitter": tool_obj.twitter,
                "facebook": tool_obj.facebook,
                "linkedin": tool_obj.linkedin,
                "tiktok": tool_obj.tiktok,
                "youtube": tool_obj.youtube,
                "instagram": tool_obj.instagram,
                "price_from": tool_obj.price_from,
                "price_to": tool_obj.price_to,
                "created_at": tool_obj.created_at.isoformat()
                if tool_obj.created_at
                else "",
                "updated_at": tool_obj.updated_at.isoformat()
                if tool_obj.updated_at
                else "",
            }

        except Exception as e:
            logger.error(f"❌ Error fetching tool from PostgreSQL: {e}")
            return None

    def _merge_tool_data(
        self, pinecone_tool: Dict[str, Any], postgres_tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge Pinecone tool data with PostgreSQL tool data.

        Args:
            pinecone_tool: Tool data from Pinecone
            postgres_tool: Tool data from PostgreSQL

        Returns:
            Merged tool data with PostgreSQL data taking priority
        """
        try:
            # Start with Pinecone data
            merged_tool = pinecone_tool.copy()

            # Override with PostgreSQL data (more reliable and detailed)
            merged_tool.update(
                {
                    "ID": postgres_tool["id"],
                    "Title": postgres_tool["title"],
                    "Description": postgres_tool["description"],  # More detailed
                    "Category": postgres_tool["category"],
                    "Features": postgres_tool["features"],  # More detailed
                    "Tags (Keywords)": postgres_tool["tags"],
                    "Website": postgres_tool["website"],
                    "Twitter": postgres_tool["twitter"],
                    "Facebook": postgres_tool["facebook"],
                    "LinkedIn": postgres_tool["linkedin"],
                    "TikTok": postgres_tool["tiktok"],
                    "YouTube": postgres_tool["youtube"],
                    "Instagram": postgres_tool["instagram"],
                    "Price From": postgres_tool["price_from"],
                    "Price To": postgres_tool["price_to"],
                    "Source": "PostgreSQL Database (Enriched)",
                    "_postgres_enriched": True,  # Flag to indicate enrichment
                    "_detailed_description": postgres_tool["description"],
                    "_detailed_features": postgres_tool["features"],
                }
            )

            # Preserve Pinecone-specific fields
            if "Similarity Score" in pinecone_tool:
                merged_tool["Similarity Score"] = pinecone_tool["Similarity Score"]

            return merged_tool

        except Exception as e:
            logger.error(f"❌ Error merging tool data: {e}")
            return pinecone_tool  # Return original on error

    def _clean_website_url(self, website: str) -> str:
        """
        Clean website URL for comparison.

        Args:
            website: Raw website URL

        Returns:
            Cleaned website URL
        """
        try:
            if not website:
                return ""

            # Remove protocol and www
            cleaned = (
                website.lower()
                .replace("https://", "")
                .replace("http://", "")
                .replace("www.", "")
                .strip()
                .rstrip("/")
            )

            # Extract domain name
            if "/" in cleaned:
                cleaned = cleaned.split("/")[0]

            return cleaned

        except Exception:
            return ""

    async def get_tools_with_websites_only(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter tools to only include those with valid websites.

        Args:
            tools: List of tools to filter

        Returns:
            List of tools with valid websites only
        """
        try:
            filtered_tools = []

            for tool in tools:
                website = tool.get("Website") or tool.get("website", "")

                # Check if website is valid
                if self._is_valid_website(website):
                    filtered_tools.append(tool)
                else:
                    title = tool.get("Title", "Unknown")
                    logger.debug(f"❌ Filtered out tool '{title}' - no valid website")

            logger.info(
                f"🔍 Filtered {len(tools)} tools → {len(filtered_tools)} tools with websites"
            )
            return filtered_tools

        except Exception as e:
            logger.error(f"❌ Error filtering tools by website: {e}")
            return tools  # Return original tools on error

    def _is_valid_website(self, website: str) -> bool:
        """
        Check if a website URL is valid.

        Args:
            website: Website URL to validate

        Returns:
            True if website is valid, False otherwise
        """
        try:
            if not website or not isinstance(website, str):
                return False

            website = website.strip()

            # Must start with http:// or https://
            if not (website.startswith("http://") or website.startswith("https://")):
                return False

            # Must have a domain
            if len(website) < 10:  # Minimum length for a valid URL
                return False

            # Must not be example.com or placeholder
            invalid_domains = [
                "example.com",
                "placeholder.com",
                "test.com",
                "demo.com",
                "sample.com",
            ]
            # Check if website contains any invalid domains
            return all(
                invalid_domain.lower() not in website.lower()
                for invalid_domain in invalid_domains
            )

        except Exception:
            return False

    # fix: invalid_domain is a string, so we need to convert it to lowercase
    async def search_postgres_tools(
        self, query: str, max_results: int = 10, min_similarity: float = 0.045
    ) -> List[Dict[str, Any]]:
        """
        Search tools directly in PostgreSQL database.

        Args:
            query: Search query
            max_results: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of tools from PostgreSQL search
        """
        try:
            from management.tools.models import Tool

            logger.info(f"🔍 Searching PostgreSQL for: {query}")

            # Use the Tool model's search method
            search_results = await sync_to_async(
                lambda: list(
                    Tool.search(query)[: max_results * 2]
                )  # Get more to allow filtering
            )()

            # Convert to dictionary format and filter by similarity
            tools = []
            for tool_obj in search_results:
                similarity = getattr(tool_obj, "total_similarity", 0)

                if similarity >= min_similarity:
                    tool_dict = {
                        "ID": str(tool_obj.id),
                        "Title": tool_obj.title,
                        "Description": tool_obj.description,
                        "Category": tool_obj.category,
                        "Features": tool_obj.features,
                        "Tags (Keywords)": tool_obj.tags,
                        "Website": tool_obj.website,
                        "Twitter": tool_obj.twitter,
                        "Facebook": tool_obj.facebook,
                        "LinkedIn": tool_obj.linkedin,
                        "TikTok": tool_obj.tiktok,
                        "YouTube": tool_obj.youtube,
                        "Instagram": tool_obj.instagram,
                        "Price From": tool_obj.price_from,
                        "Price To": tool_obj.price_to,
                        "Source": "PostgreSQL Database",
                        "Similarity Score": similarity,
                        "_postgres_native": True,
                    }
                    tools.append(tool_dict)

            # Filter to only include tools with websites
            tools_with_websites = await self.get_tools_with_websites_only(tools)

            # Limit results
            final_tools = tools_with_websites[:max_results]

            logger.info(
                f"✅ PostgreSQL search found {len(final_tools)} tools "
                f"(filtered from {len(search_results)} total)"
            )
            return final_tools

        except Exception as e:
            logger.error(f"❌ Error searching PostgreSQL: {e}")
            return []

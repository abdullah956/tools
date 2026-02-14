"""Pinecone vector database service for AI tool search."""

import logging
import os
from typing import Any, Dict, List

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from ai_tool_recommender.ai_agents.tools.pinecone.query_helper import (
    PineconeQueryHelper,
)
from ai_tool_recommender.ai_agents.tools.postgres_tool_service import (
    PostgresToolService,
)

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for searching AI tools using Pinecone vector database."""

    def __init__(self):
        """Initialize the Pinecone service."""
        self.pinecone_client = None
        self.index = None
        self.scraping_index = None  # NEW: Index for scraped tools
        self.embeddings = None
        self.postgres_service = PostgresToolService()  # NEW: PostgreSQL integration
        self._initialize_services()

    def _initialize_services(self):
        """Initialize Pinecone client and embeddings."""
        try:
            # Initialize Pinecone client
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if pinecone_api_key:
                self.pinecone_client = Pinecone(api_key=pinecone_api_key)
                logger.info("Pinecone client initialized successfully")

                # Connect to index
                self._connect_to_index()
            else:
                logger.warning("PINECONE_API_KEY not found in environment variables")

            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings()
            logger.info("OpenAI embeddings initialized")

        except Exception as e:
            logger.error(f"Error initializing Pinecone services: {e}")

    def _connect_to_index(self):
        """Connect to Pinecone indexes (both old and new scraping index)."""
        try:
            # Connect to ONLY the NEW scraping index (all tools are here now)
            scraping_index_name = os.getenv(
                "PINECONE_TOOL_INDEX", "scraping-tool-index"
            )

            # Use both index and scraping_index for backward compatibility
            self.index = self.pinecone_client.Index(scraping_index_name)
            self.scraping_index = self.index  # Point to same index

            stats = self.index.describe_index_stats()
            logger.info(
                f"✅ Connected to Pinecone scraping index '{scraping_index_name}'"
            )
            logger.info(f"Index stats: {stats}")

        except Exception as e:
            logger.error(f"Error connecting to Pinecone indexes: {e}")
            self.index = None
            self.scraping_index = None

    def _normalize_tool_name(self, name: str) -> str:
        """Normalize tool name for matching (lowercase, strip, remove special chars).

        Args:
            name: Tool name to normalize

        Returns:
            Normalized tool name
        """
        if not name:
            return ""
        # Lowercase, strip, remove extra spaces
        normalized = name.lower().strip()
        # Remove common suffixes/prefixes that might cause mismatches
        normalized = normalized.replace(" - ", " ").replace(" | ", " ")
        return normalized

    async def search_tools_exact_match(
        self, tool_names: List[str], max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search Pinecone for exact tool name matches using metadata filters.

        This method searches for tools by exact title match first, then falls back
        to semantic search if exact matches are not found.

        Args:
            tool_names: List of exact tool names to search for
            max_results: Maximum number of results to return per tool

        Returns:
            List of AI tool data matching exact tool names
        """
        if not self.index or not tool_names:
            logger.warning("Pinecone index not available or no tool names provided")
            return []

        try:
            logger.info(f"🎯 Searching Pinecone for exact tool matches: {tool_names}")
            all_tools = []
            seen_tool_ids = set()  # Track by tool ID to avoid duplicates

            # Normalize tool names for matching
            normalized_names = [self._normalize_tool_name(name) for name in tool_names]

            # Strategy 1: Try exact metadata filter match
            # Note: Pinecone metadata filters require exact matches
            # We'll search with embeddings but filter results by title
            if self.embeddings:
                # Generate embedding for the combined tool names query
                combined_query = " ".join(tool_names)
                query_embedding = PineconeQueryHelper.prepare_query_embedding(
                    combined_query, self.embeddings
                )

                if query_embedding and self.index:
                    try:
                        # Search with higher top_k to get more candidates for filtering
                        results = self.index.query(
                            vector=query_embedding,
                            top_k=max_results
                            * len(tool_names)
                            * 2,  # Get more for filtering
                            include_values=True,
                            include_metadata=True,
                            namespace="",
                        )

                        # Process and filter for exact matches
                        for match in results.matches:
                            try:
                                metadata = match.metadata
                                tool_title = metadata.get("title", "").strip()
                                tool_id = metadata.get("ID", "")

                                # Skip if already seen (by ID)
                                if tool_id and tool_id in seen_tool_ids:
                                    continue

                                # Check if tool title matches any of the requested tool names
                                normalized_title = self._normalize_tool_name(tool_title)
                                is_exact_match = False

                                for normalized_name in normalized_names:
                                    # Exact match (normalized)
                                    if normalized_title == normalized_name:
                                        is_exact_match = True
                                        break
                                    # Partial match (tool name is in title or vice versa)
                                    # Additional check: ensure it's not a false positive
                                    # (e.g., "Slack" shouldn't match "Slackbot")
                                    if (
                                        normalized_name in normalized_title
                                        or normalized_title in normalized_name
                                    ) and len(
                                        normalized_name
                                    ) >= 4:  # Only for names >= 4 chars
                                        is_exact_match = True
                                        break

                                if is_exact_match:
                                    tool = {
                                        "Title": tool_title,
                                        "Description": metadata.get("description", "")[
                                            :500
                                        ],
                                        "Website": metadata.get("website", ""),
                                        "Category": metadata.get("category", ""),
                                        "Master Category": metadata.get(
                                            "master_category", ""
                                        ),
                                        "Similarity Score": match.score,
                                        "Source": "Scraped Tool Database (Exact Match)",
                                        "ID": tool_id,
                                        "Features": metadata.get("features", ""),
                                        "_match_type": "exact",
                                    }
                                    all_tools.append(tool)
                                    if tool_id:
                                        seen_tool_ids.add(tool_id)

                                    # Stop if we have enough results
                                    if len(all_tools) >= max_results * len(tool_names):
                                        break

                            except Exception as e:
                                logger.error(
                                    f"Error processing exact match result: {e}"
                                )
                                continue

                    except Exception as e:
                        logger.error(f"Error querying Pinecone for exact match: {e}")

            # Sort by similarity score (highest first)
            all_tools.sort(key=lambda x: x.get("Similarity Score", 0), reverse=True)

            # Deduplicate by website
            filtered_tools = []
            seen_websites = set()
            for tool in all_tools:
                website = tool.get("Website", "").lower().strip()
                if website and website in seen_websites:
                    continue
                if website:
                    seen_websites.add(website)
                filtered_tools.append(tool)

            logger.info(
                f"✅ Found {len(filtered_tools)} exact tool matches from Pinecone"
            )
            return filtered_tools[: max_results * len(tool_names)]

        except Exception as e:
            logger.error(f"Error in exact match search: {e}")
            return []

    async def search_tools(
        self, query: str, max_results: int = 10, similarity_threshold: float = 0.045
    ) -> List[Dict[str, Any]]:
        """Search for AI tools using BOTH Pinecone indexes (old + new scraping).

        Args:
            query: The search query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results

        Returns:
            List of AI tool data from both Pinecone indexes
        """
        if not self.embeddings:
            logger.error("Embeddings not initialized")
            return []

        try:
            logger.info(
                f"🔍 [PINECONE] Starting search for: '{query}' (max={max_results})"
            )

            # Generate embedding using query helper
            query_embedding = PineconeQueryHelper.prepare_query_embedding(
                query, self.embeddings
            )
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            logger.info(
                f"✅ Query embedding generated with dimension: {len(query_embedding)}"
            )

            all_tools = []

            # Search the scraping index (all tools are here now)
            if self.index:
                logger.info("Querying scraping index")
                try:
                    results = self.index.query(
                        vector=query_embedding,
                        top_k=max_results * 2,  # Get more to allow for filtering
                        include_values=True,
                        include_metadata=True,
                        namespace="",  # Scraping index uses default namespace
                    )
                    logger.info(
                        f"Scraping index returned {len(results.matches)} matches"
                    )

                    # Process scraping results - convert to same format
                    all_tools = self._process_scraping_results(results)
                except Exception as e:
                    logger.error(f"Error querying scraping index: {e}")

            # Filter by similarity threshold and intelligent deduplication
            print(
                f"🔍 Filtering {len(all_tools)} tools by similarity threshold {similarity_threshold}"
            )
            filtered_tools = []
            seen_identifiers = set()

            # Sort by similarity score (highest first)
            all_tools.sort(key=lambda x: x.get("Similarity Score", 0), reverse=True)

            for i, tool in enumerate(all_tools):
                similarity_score = tool.get("Similarity Score", 0.8)

                # Skip if below threshold
                if similarity_score < similarity_threshold:
                    continue

                # Create multiple identifiers for intelligent deduplication
                identifiers = self._create_tool_identifiers_for_dedup(tool)

                # Check if any identifier already exists
                is_duplicate = False
                for identifier in identifiers:
                    if identifier in seen_identifiers:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                # Add all identifiers to seen set
                seen_identifiers.update(identifiers)
                filtered_tools.append(tool)

                # Stop if we have enough results
                if len(filtered_tools) >= max_results:
                    break

            # Step 4: Enrich with PostgreSQL data for better LLM selection
            enriched_tools = (
                await self.postgres_service.enrich_tools_with_postgres_data(
                    filtered_tools
                )
            )

            # Step 5: Filter to only include tools with valid websites
            final_tools = await self.postgres_service.get_tools_with_websites_only(
                enriched_tools
            )

            logger.info(
                f"Pinecone search completed. Found {len(final_tools)} tools with websites "
                f"(enriched from {len(filtered_tools)}, filtered from {len(all_tools)} by similarity threshold {similarity_threshold})"
            )
            return final_tools

        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            return []

    def _process_scraping_results(self, results) -> List[Dict[str, Any]]:
        """Process results from the NEW scraping index and convert to standard format.

        The scraping index has a different metadata structure:
        - ID: UUID of the tool
        - title: Tool name
        - category: Tool category
        - master_category: Master category
        - description: Tool description
        - website: Tool website
        - text: Chunk text (not needed for search results)

        Args:
            results: Pinecone query results from scraping index

        Returns:
            List of tools in standard format
        """
        tools = []

        for match in results.matches:
            try:
                metadata = match.metadata
                score = match.score

                # Convert scraping metadata to standard tool format
                tool = {
                    "Title": metadata.get("title", ""),
                    "Description": metadata.get("description", "")[
                        :500
                    ],  # Limit description
                    "Website": metadata.get("website", ""),
                    "Category": metadata.get("category", ""),
                    "Master Category": metadata.get("master_category", ""),
                    "Similarity Score": score,
                    "Source": "Scraped Tool Database",
                    "ID": metadata.get("ID", ""),  # UUID from CombinedText
                    # Additional fields that might be in the scraping metadata
                    "Features": metadata.get("features", ""),
                }

                tools.append(tool)

            except Exception as e:
                logger.error(f"Error processing scraping result: {e}")
                continue

        return tools

    async def add_tool(self, tool_data: Dict[str, Any]) -> bool:
        """Add a single tool to Pinecone using the same approach as Excel handler.

        Includes deduplication check before adding.

        Args:
            tool_data: Dictionary containing tool information

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return False

            # Check for duplicates before adding
            if await self._is_duplicate_tool(tool_data):
                logger.info(
                    f"Tool already exists, skipping: {tool_data.get('Title', 'Unknown')}"
                )
                return False

            # Get description for embedding
            description = tool_data.get("Description", "").strip()
            if not description:
                logger.warning("No description found for tool, using title")
                description = tool_data.get("Title", "AI Tool").strip()

            if not description:
                logger.error("No description or title found for tool")
                return False

            # Generate embedding using the same method as Excel handler
            from ai_tool_recommender.utils.embeddings import get_embedding

            embedding = await get_embedding(description)

            # Generate unique ID
            import uuid

            record_id = str(uuid.uuid4())

            # Prepare vector for Pinecone (same format as Excel handler)
            vector_data = {"id": record_id, "values": embedding, "metadata": tool_data}

            # Upsert to Pinecone (same method as Excel handler)
            import asyncio

            await asyncio.to_thread(
                self.index.upsert,
                vectors=[vector_data],
                namespace=os.getenv("PINECONE_NAMESPACE"),
            )

            logger.info(
                f"Successfully added tool to Pinecone: {tool_data.get('Title', 'Unknown')}"
            )
            return True

        except Exception as e:
            logger.error(f"Error adding tool to Pinecone: {e}")
            return False

    async def _is_duplicate_tool(self, tool_data: Dict[str, Any]) -> bool:
        """Check if a tool is a duplicate before adding to Pinecone.

        Args:
            tool_data: Tool data to check

        Returns:
            True if duplicate, False otherwise
        """
        try:
            title = tool_data.get("Title", "").strip().lower()
            website = tool_data.get("Website", "").strip().lower()

            if not title and not website:
                return False

            # Search for existing tools
            search_query = f"{title} {website}".strip()
            if not search_query:
                return False

            # Use lower similarity threshold for duplicate detection
            existing_tools = await self.search_tools(
                query=search_query, max_results=5, similarity_threshold=0.6
            )

            if not existing_tools:
                return False

            # Check for duplicates
            for existing_tool in existing_tools:
                existing_title = existing_tool.get("Title", "").strip().lower()
                existing_website = existing_tool.get("Website", "").strip().lower()

                # Exact matches
                if title == existing_title or website == existing_website:
                    return True

                # Domain matches
                if website and existing_website:
                    try:
                        from urllib.parse import urlparse

                        current_domain = urlparse(website).netloc.lower()
                        existing_domain = urlparse(existing_website).netloc.lower()

                        if (
                            current_domain
                            and existing_domain
                            and current_domain == existing_domain
                        ):
                            return True
                    except Exception:
                        pass

            return False

        except Exception as e:
            logger.error(f"Error checking for duplicate tool: {e}")
            return False

    async def add_tools_batch(self, tools_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add multiple tools to Pinecone in batch (similar to Excel handler).

        Includes deduplication check for each tool.

        Args:
            tools_data: List of tool dictionaries

        Returns:
            Dictionary with success/failure counts
        """
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return {"success": 0, "failed": 0, "total": len(tools_data)}

            vectors = []
            successful_count = 0
            duplicate_count = 0

            for tool_data in tools_data:
                try:
                    # Check for duplicates before processing
                    if await self._is_duplicate_tool(tool_data):
                        duplicate_count += 1
                        logger.info(
                            f"Skipping duplicate tool: {tool_data.get('Title', 'Unknown')}"
                        )
                        continue

                    # Get description for embedding
                    description = tool_data.get("Description", "").strip()
                    if not description:
                        description = tool_data.get("Title", "AI Tool").strip()

                    if not description:
                        logger.warning(
                            f"Skipping tool without description: {tool_data.get('Title', 'Unknown')}"
                        )
                        continue

                    # Generate embedding
                    from ai_tool_recommender.utils.embeddings import get_embedding

                    embedding = await get_embedding(description)

                    # Generate unique ID
                    import uuid

                    record_id = str(uuid.uuid4())

                    # Prepare vector
                    vector_data = {
                        "id": record_id,
                        "values": embedding,
                        "metadata": tool_data,
                    }

                    vectors.append(vector_data)
                    successful_count += 1

                except Exception as e:
                    logger.error(
                        f"Error preparing tool {tool_data.get('Title', 'Unknown')}: {e}"
                    )
                    continue

            if vectors:
                # Upsert all vectors at once (same as Excel handler)
                import asyncio

                await asyncio.to_thread(
                    self.index.upsert,
                    vectors=vectors,
                    namespace=os.getenv("PINECONE_NAMESPACE"),
                )

                logger.info(
                    f"Successfully added {len(vectors)} tools to Pinecone in batch (skipped {duplicate_count} duplicates)"
                )

            return {
                "success": len(vectors),
                "failed": len(tools_data) - len(vectors) - duplicate_count,
                "duplicates": duplicate_count,
                "total": len(tools_data),
            }

        except Exception as e:
            logger.error(f"Error adding tools batch to Pinecone: {e}")
            return {
                "success": 0,
                "failed": len(tools_data),
                "duplicates": 0,
                "total": len(tools_data),
            }

    async def add_tools_batch_detailed(
        self, tools_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add multiple tools to Pinecone with detailed status tracking for each tool.

        Args:
            tools_data: List of tool dictionaries

        Returns:
            Dictionary with detailed results including individual tool status
        """
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return {
                    "success": 0,
                    "failed": len(tools_data),
                    "duplicates": 0,
                    "skipped": 0,
                    "total": len(tools_data),
                    "details": [],
                }

            vectors = []
            details = []
            successful_count = 0
            duplicate_count = 0
            failed_count = 0
            skipped_count = 0

            for tool_data in tools_data:
                title = tool_data.get("Title", "Unknown")
                website = tool_data.get("Website", "")

                try:
                    # Check for duplicates before processing
                    duplicate_check = await self._check_duplicate_detailed(tool_data)

                    if duplicate_check["is_duplicate"]:
                        duplicate_count += 1
                        details.append(
                            {
                                "title": title,
                                "website": website,
                                "status": "duplicate",
                                "reason": duplicate_check["reason"],
                                "similarity_score": duplicate_check.get(
                                    "similarity_score", 0.0
                                ),
                            }
                        )
                        logger.info(f"Skipping duplicate tool: {title}")
                        continue

                    # Get description for embedding
                    description = tool_data.get("Description", "").strip()
                    if not description:
                        description = tool_data.get("Title", "AI Tool").strip()

                    if not description:
                        skipped_count += 1
                        details.append(
                            {
                                "title": title,
                                "website": website,
                                "status": "skipped",
                                "reason": "No description or title found",
                                "similarity_score": 0.0,
                            }
                        )
                        logger.warning(f"Skipping tool without description: {title}")
                        continue

                    # Generate embedding
                    from ai_tool_recommender.utils.embeddings import get_embedding

                    embedding = await get_embedding(description)

                    # Generate unique ID
                    import uuid

                    record_id = str(uuid.uuid4())

                    # Prepare vector
                    vector_data = {
                        "id": record_id,
                        "values": embedding,
                        "metadata": tool_data,
                    }

                    vectors.append(vector_data)
                    successful_count += 1

                    details.append(
                        {
                            "title": title,
                            "website": website,
                            "status": "added",
                            "reason": "Successfully processed and added",
                            "similarity_score": 0.0,
                        }
                    )

                except Exception as e:
                    failed_count += 1
                    details.append(
                        {
                            "title": title,
                            "website": website,
                            "status": "failed",
                            "reason": f"Error processing tool: {str(e)}",
                            "similarity_score": 0.0,
                        }
                    )
                    logger.error(f"Error preparing tool {title}: {e}")
                    continue

            if vectors:
                # Upsert all vectors at once
                import asyncio

                await asyncio.to_thread(
                    self.index.upsert,
                    vectors=vectors,
                    namespace=os.getenv("PINECONE_NAMESPACE"),
                )

                logger.info(
                    f"Successfully added {len(vectors)} tools to Pinecone in batch (skipped {duplicate_count} duplicates, {skipped_count} skipped)"
                )

            return {
                "success": len(vectors),
                "failed": failed_count,
                "duplicates": duplicate_count,
                "skipped": skipped_count,
                "total": len(tools_data),
                "details": details,
            }

        except Exception as e:
            logger.error(f"Error adding tools batch to Pinecone: {e}")
            return {
                "success": 0,
                "failed": len(tools_data),
                "duplicates": 0,
                "skipped": 0,
                "total": len(tools_data),
                "details": [],
            }

    async def _check_duplicate_detailed(
        self, tool_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if a tool is a duplicate with detailed information.

        Args:
            tool_data: Tool data to check

        Returns:
            Dictionary with duplicate status and details
        """
        try:
            title = tool_data.get("Title", "").strip().lower()
            website = tool_data.get("Website", "").strip().lower()

            if not title and not website:
                return {
                    "is_duplicate": False,
                    "reason": "No title or website to check",
                    "similarity_score": 0.0,
                }

            # Search for existing tools
            search_query = f"{title} {website}".strip()
            if not search_query:
                return {
                    "is_duplicate": False,
                    "reason": "Empty search query",
                    "similarity_score": 0.0,
                }

            # Use lower similarity threshold for duplicate detection
            existing_tools = await self.search_tools(
                query=search_query, max_results=5, similarity_threshold=0.6
            )

            if not existing_tools:
                return {
                    "is_duplicate": False,
                    "reason": "No similar tools found",
                    "similarity_score": 0.0,
                }

            # Check for duplicates with detailed matching
            for existing_tool in existing_tools:
                existing_title = existing_tool.get("Title", "").strip().lower()
                existing_website = existing_tool.get("Website", "").strip().lower()
                similarity_score = existing_tool.get("Similarity Score", 0.0)

                # Exact matches
                if title == existing_title:
                    return {
                        "is_duplicate": True,
                        "reason": f"Exact title match with '{existing_title}'",
                        "similarity_score": similarity_score,
                    }

                if website == existing_website:
                    return {
                        "is_duplicate": True,
                        "reason": f"Exact website match with '{existing_website}'",
                        "similarity_score": similarity_score,
                    }

                # Domain matches
                if website and existing_website:
                    try:
                        from urllib.parse import urlparse

                        current_domain = urlparse(website).netloc.lower()
                        existing_domain = urlparse(existing_website).netloc.lower()

                        if (
                            current_domain
                            and existing_domain
                            and current_domain == existing_domain
                        ):
                            return {
                                "is_duplicate": True,
                                "reason": f"Same domain '{current_domain}' as existing tool",
                                "similarity_score": similarity_score,
                            }
                    except Exception:
                        pass

                # High similarity match
                if similarity_score > 0.8:
                    return {
                        "is_duplicate": True,
                        "reason": f"High similarity ({similarity_score:.2f}) with '{existing_title}'",
                        "similarity_score": similarity_score,
                    }

            return {
                "is_duplicate": False,
                "reason": "No duplicates found",
                "similarity_score": 0.0,
            }

        except Exception as e:
            logger.error(f"Error checking for duplicate tool: {e}")
            return {
                "is_duplicate": False,
                "reason": f"Error during duplicate check: {str(e)}",
                "similarity_score": 0.0,
            }

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index.

        Returns:
            Dictionary with index statistics
        """
        try:
            if not self.index:
                return {"error": "Pinecone index not available"}

            import asyncio

            # Get index stats
            stats = await asyncio.to_thread(self.index.describe_index_stats)

            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "namespaces": stats.get("namespaces", {}),
                "status": "connected",
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e), "status": "error"}

    def _create_tool_identifiers_for_dedup(self, tool: Dict[str, Any]) -> List[str]:
        """Create multiple identifiers for intelligent deduplication."""
        identifiers = []

        try:
            # Get tool data (handle both formats)
            title = tool.get("title", tool.get("Title", "")).strip()
            website = tool.get("website", tool.get("Website", "")).strip()

            # 1. Website-based identifier (most reliable)
            if website:
                clean_website = (
                    website.lower()
                    .replace("https://", "")
                    .replace("http://", "")
                    .replace("www.", "")
                    .rstrip("/")
                )
                if clean_website and len(clean_website) > 5:
                    identifiers.append(f"website:{clean_website}")

                    # Also add domain-only identifier
                    try:
                        from urllib.parse import urlparse

                        domain = urlparse(website).netloc.lower()
                        if domain:
                            identifiers.append(f"domain:{domain}")
                    except Exception:
                        pass

            # 2. Title-based identifier
            if title and len(title) > 3:
                # Exact title
                identifiers.append(f"title:{title.lower()}")

                # Normalized title (remove common variations)
                normalized = (
                    title.lower()
                    .replace(" ai", "")
                    .replace("ai ", "")
                    .replace(".", "")
                    .replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )
                if len(normalized) > 3:
                    identifiers.append(f"normalized:{normalized}")

            # 3. Fallback identifier
            if not identifiers:
                identifiers.append(f"fallback:{title.lower()}" if title else "unknown")

            return identifiers

        except Exception as e:
            logger.error(f"Error creating tool identifiers: {e}")
            return [f"error:{tool.get('title', tool.get('Title', 'unknown'))}"]

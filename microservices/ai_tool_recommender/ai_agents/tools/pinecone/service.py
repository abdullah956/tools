"""Pinecone vector database service for AI tool search."""

import logging
import os
from typing import Any, Dict, List

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from microservices.ai_tool_recommender.ai_agents.tools.pinecone.query_helper import (
    PineconeQueryHelper,
)

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for searching AI tools using Pinecone vector database."""

    def __init__(self):
        """Initialize the Pinecone service."""
        self.pinecone_client = None
        self.index = None
        self.embeddings = None
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
        """Connect to Pinecone index."""
        try:
            index_name = os.getenv("PINECONE_TOOL_INDEX", "ai-tools-index")
            self.index = self.pinecone_client.Index(index_name)

            # Get index stats for debugging
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index '{index_name}'")
            logger.info(f"Index stats: {stats}")

        except Exception as e:
            logger.error(f"Error connecting to Pinecone index: {e}")
            self.index = None

    async def search_tools(
        self, query: str, max_results: int = 10, similarity_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Search for AI tools using Pinecone vector database.

        Args:
            query: The search query
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results

        Returns:
            List of AI tool data from Pinecone
        """
        if not self.index or not self.embeddings:
            logger.error("Pinecone index or embeddings not initialized")
            return []

        try:
            print(f"🔍 Searching Pinecone for tools with query: {query}")
            logger.info(f"Searching Pinecone for tools with query: {query}")

            # Generate embedding using query helper
            print("🔍 Generating query embedding...")
            query_embedding = PineconeQueryHelper.prepare_query_embedding(
                query, self.embeddings
            )
            if not query_embedding:
                print("❌ Failed to generate query embedding")
                logger.error("Failed to generate query embedding")
                return []
            print(f"✅ Query embedding generated with dimension: {len(query_embedding)}")

            # Query Pinecone without namespace (default namespace)
            print("🔍 Querying Pinecone without namespace (using default namespace)")
            logger.info("Querying Pinecone without namespace (using default namespace)")
            results = self.index.query(
                vector=query_embedding,
                top_k=max_results,
                include_values=True,
                include_metadata=True,
            )
            print(f"✅ Pinecone query returned {len(results.matches)} matches")
            logger.info(f"Pinecone query returned {len(results.matches)} matches")

            # Process results using query helper
            print(f"🔍 Processing {len(results.matches)} matches...")
            tools = PineconeQueryHelper.process_search_results(results, None)
            print(f"✅ Processed {len(tools)} tools from results")

            # Filter by similarity threshold
            print(
                f"🔍 Filtering {len(tools)} tools by similarity threshold {similarity_threshold}"
            )
            filtered_tools = []
            for i, tool in enumerate(tools):
                # Get similarity score from metadata if available
                similarity_score = tool.get("Similarity Score", 0.8)
                print(
                    f"🔍 Tool {i + 1}: similarity_score={similarity_score}, threshold={similarity_threshold}"
                )
                if similarity_score >= similarity_threshold:
                    filtered_tools.append(tool)
                    print(f"✅ Tool {i + 1} passed filter")
                else:
                    print(f"❌ Tool {i + 1} filtered out (score too low)")

            print(f"✅ Final result: {len(filtered_tools)} tools after filtering")
            logger.info(
                f"Pinecone search completed. Found {len(filtered_tools)} tools (filtered from {len(tools)} by similarity threshold {similarity_threshold})"
            )
            return filtered_tools

        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            return []

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
            from microservices.ai_tool_recommender.utils.embeddings import get_embedding

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
                    from microservices.ai_tool_recommender.utils.embeddings import (
                        get_embedding,
                    )

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
                    from microservices.ai_tool_recommender.utils.embeddings import (
                        get_embedding,
                    )

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

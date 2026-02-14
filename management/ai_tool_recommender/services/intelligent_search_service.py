"""Intelligent Search Service - Implements the new search architecture.

This service implements the improved search flow:
1. Search Pinecone with refined query
2. If meaningful tools found, skip internet search
3. Enrich Pinecone tools with Postgres data
4. Select optimal tools
"""

import logging
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.tools.internet_search import InternetSearchService
from ai_tool_recommender.ai_agents.tools.pinecone import PineconeService
from ai_tool_recommender.ai_agents.tools.postgres_tool_service import (
    PostgresToolService,
)

logger = logging.getLogger(__name__)


class IntelligentSearchService:
    """Service that implements the intelligent search architecture."""

    def __init__(self):
        """Initialize the intelligent search service."""
        self.pinecone_service = PineconeService()
        self.postgres_service = PostgresToolService()
        self.internet_service = InternetSearchService()
        self.llm = get_shared_llm()

    async def search_tools_intelligently(
        self,
        refined_query: str,
        search_queries: List[Dict[str, Any]],
        max_results: int = 10,
        meaningfulness_threshold: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute intelligent search with Pinecone-first approach.

        Args:
            refined_query: The comprehensive refined query
            search_queries: List of search query objects from query generator
            max_results: Maximum results to return
            meaningfulness_threshold: Minimum tools needed from Pinecone to skip internet

        Returns:
            Dict with search results and metadata
        """
        try:
            logger.info("🎯 Starting intelligent search with Pinecone-first approach")

            # Step 1: Search Pinecone with all search queries
            pinecone_tools = await self._search_pinecone_parallel(search_queries)
            logger.info(f"📦 Pinecone search found {len(pinecone_tools)} tools")

            # Step 2: Check if Pinecone results are meaningful
            is_meaningful = await self._are_results_meaningful(
                pinecone_tools, refined_query, meaningfulness_threshold
            )

            internet_tools = []
            if is_meaningful:
                logger.info(
                    f"✅ Pinecone results are meaningful ({len(pinecone_tools)} tools), skipping internet search"
                )
            else:
                logger.info(
                    f"⚠️ Pinecone results insufficient ({len(pinecone_tools)} tools), performing internet search"
                )
                # Step 3: Perform internet search if Pinecone results are insufficient
                internet_tools = await self._search_internet_parallel(search_queries)
                logger.info(f"🌐 Internet search found {len(internet_tools)} tools")

            # Step 4: Combine and deduplicate tools
            all_tools = self._combine_and_deduplicate(pinecone_tools, internet_tools)
            logger.info(f"📊 Combined total: {len(all_tools)} unique tools")

            # Step 5: Enrich Pinecone tools with Postgres data (using tool IDs)
            enriched_tools = await self._enrich_pinecone_tools(all_tools)
            logger.info(f"💎 Enriched {len(enriched_tools)} tools with Postgres data")

            return {
                "status": "success",
                "tools": enriched_tools,
                "metadata": {
                    "pinecone_count": len(pinecone_tools),
                    "internet_count": len(internet_tools),
                    "total_count": len(enriched_tools),
                    "skipped_internet": is_meaningful,
                    "search_queries_used": len(search_queries),
                },
            }

        except Exception as e:
            logger.error(f"❌ Error in intelligent search: {e}", exc_info=True)
            return {
                "status": "error",
                "tools": [],
                "metadata": {"error": str(e)},
            }

    async def _search_pinecone_parallel(
        self, search_queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Search Pinecone with multiple queries in parallel.

        Args:
            search_queries: List of search query objects

        Returns:
            List of tools found in Pinecone
        """
        try:
            import asyncio

            all_tools = []
            seen_ids = set()

            # Create search tasks for each query
            tasks = []
            for query_obj in search_queries:
                query_text = query_obj.get("query", "")
                if query_text:
                    task = self.pinecone_service.search_tools(
                        query_text,
                        max_results=5,  # 5 results per query
                        similarity_threshold=0.045,
                    )
                    tasks.append(task)

            # Execute all searches in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect and deduplicate results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Search {i+1} failed: {result}")
                    continue

                if isinstance(result, list):
                    for tool in result:
                        tool_id = tool.get("ID") or tool.get("id") or tool.get("Title")
                        if tool_id and tool_id not in seen_ids:
                            seen_ids.add(tool_id)
                            all_tools.append(tool)

            logger.info(
                f"✅ Pinecone parallel search: {len(all_tools)} unique tools from {len(search_queries)} queries"
            )
            return all_tools

        except Exception as e:
            logger.error(f"❌ Error in parallel Pinecone search: {e}")
            return []

    async def _are_results_meaningful(
        self, tools: List[Dict[str, Any]], refined_query: str, threshold: int
    ) -> bool:
        """
        Determine if Pinecone results are meaningful enough to skip internet search.

        Args:
            tools: List of tools from Pinecone
            refined_query: The refined query for context
            threshold: Minimum number of tools needed

        Returns:
            True if results are meaningful, False otherwise
        """
        try:
            # Quick check: If we have fewer tools than threshold, not meaningful
            if len(tools) < threshold:
                logger.info(
                    f"❌ Only {len(tools)} tools found, below threshold of {threshold}"
                )
                return False

            # Use LLM to assess quality and relevance
            tools_summary = "\n".join(
                [
                    f"- {tool.get('Title', 'Unknown')}: {tool.get('Description', '')[:100]}"
                    for tool in tools[:10]
                ]
            )

            prompt = f"""Analyze if these search results are meaningful and sufficient to solve the user's problem.

USER'S PROBLEM:
{refined_query[:500]}

SEARCH RESULTS ({len(tools)} tools found):
{tools_summary}

Evaluate:
1. Do these tools directly address the user's problem?
2. Are there enough diverse tools to build a complete solution?
3. Are the tools relevant and high-quality?

Return ONLY "YES" if results are meaningful and sufficient, or "NO" if internet search is needed.
"""

            response = await self.llm.generate_response(prompt)
            is_meaningful = "yes" in response.lower().strip()

            logger.info(
                f"🤖 LLM assessment: {'Meaningful' if is_meaningful else 'Insufficient'}"
            )
            return is_meaningful

        except Exception as e:
            logger.error(f"❌ Error assessing meaningfulness: {e}")
            # If assessment fails, be conservative and do internet search
            return len(tools) >= threshold * 2

    async def _search_internet_parallel(
        self, search_queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Search internet with multiple queries in parallel.

        Args:
            search_queries: List of search query objects

        Returns:
            List of tools found on internet
        """
        try:
            import asyncio

            all_tools = []
            seen_urls = set()

            # Limit to top 3 queries for internet search (to save time/cost)
            top_queries = search_queries[:3]

            # Create search tasks
            tasks = []
            for query_obj in top_queries:
                query_text = query_obj.get("query", "")
                if query_text:
                    task = self.internet_service.search_ai_tools(
                        query_text, max_results=3
                    )
                    tasks.append(task)

            # Execute all searches in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect and deduplicate results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Internet search {i+1} failed: {result}")
                    continue

                if isinstance(result, list):
                    for tool in result:
                        url = tool.get("Website", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_tools.append(tool)

            logger.info(
                f"✅ Internet parallel search: {len(all_tools)} unique tools from {len(top_queries)} queries"
            )
            return all_tools

        except Exception as e:
            logger.error(f"❌ Error in parallel internet search: {e}")
            return []

    def _combine_and_deduplicate(
        self, pinecone_tools: List[Dict[str, Any]], internet_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine tools from different sources and remove duplicates.

        Args:
            pinecone_tools: Tools from Pinecone
            internet_tools: Tools from internet

        Returns:
            Combined and deduplicated list
        """
        try:
            seen_identifiers = set()
            combined_tools = []

            # Add Pinecone tools first (they have priority)
            for tool in pinecone_tools:
                identifier = self._create_tool_identifier(tool)
                if identifier not in seen_identifiers:
                    seen_identifiers.add(identifier)
                    combined_tools.append(tool)

            # Add internet tools if not duplicates
            for tool in internet_tools:
                identifier = self._create_tool_identifier(tool)
                if identifier not in seen_identifiers:
                    seen_identifiers.add(identifier)
                    combined_tools.append(tool)

            logger.info(
                f"🔄 Deduplication: {len(pinecone_tools) + len(internet_tools)} → {len(combined_tools)} tools"
            )
            return combined_tools

        except Exception as e:
            logger.error(f"❌ Error combining tools: {e}")
            return pinecone_tools + internet_tools

    def _create_tool_identifier(self, tool: Dict[str, Any]) -> str:
        """Create unique identifier for deduplication."""
        try:
            # Try website first (most reliable)
            website = tool.get("Website", "").lower().strip()
            if website and website.startswith(("http://", "https://")):
                from urllib.parse import urlparse

                domain = urlparse(website).netloc
                if domain:
                    return f"domain:{domain}"

            # Fallback to normalized title
            title = tool.get("Title", "").lower().strip()
            if title:
                normalized = (
                    title.replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                    .replace(".", "")
                )
                return f"title:{normalized}"

            return f"unknown:{id(tool)}"

        except Exception:
            return f"error:{id(tool)}"

    async def _enrich_pinecone_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich Pinecone tools with Postgres data using tool IDs.

        Args:
            tools: List of tools (mixed Pinecone and internet)

        Returns:
            List of enriched tools
        """
        try:
            # Separate Pinecone tools from internet tools
            pinecone_tools = [
                tool
                for tool in tools
                if "Pinecone" in tool.get("Source", "")
                or "Scraped" in tool.get("Source", "")
            ]
            internet_tools = [
                tool for tool in tools if "Internet Search" in tool.get("Source", "")
            ]

            logger.info(
                f"📊 Enriching {len(pinecone_tools)} Pinecone tools with Postgres data"
            )

            # Enrich Pinecone tools with Postgres data
            if pinecone_tools:
                enriched_pinecone = (
                    await self.postgres_service.enrich_tools_with_postgres_data(
                        pinecone_tools
                    )
                )
            else:
                enriched_pinecone = []

            # Combine enriched Pinecone tools with internet tools (no enrichment needed)
            all_enriched = enriched_pinecone + internet_tools

            enriched_count = len(
                [t for t in all_enriched if t.get("_postgres_enriched")]
            )
            logger.info(
                f"✅ Enrichment complete: {enriched_count}/{len(pinecone_tools)} Pinecone tools enriched"
            )

            return all_enriched

        except Exception as e:
            logger.error(f"❌ Error enriching tools: {e}")
            return tools

"""Parallel Tool Search Service - Executes multiple targeted searches simultaneously."""

import asyncio
import logging
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.performance_monitor import performance_monitor
from ai_tool_recommender.ai_agents.tools.ai_tool_recommender import AIToolRecommender
from ai_tool_recommender.ai_agents.tools.internet_search import InternetSearchService
from ai_tool_recommender.ai_agents.tools.pinecone import PineconeService

logger = logging.getLogger(__name__)


class ParallelToolSearchService:
    """
    Service for executing multiple targeted tool searches in parallel.

    This service:
    - Takes multiple search queries from QueryDecomposerAgent
    - Executes Pinecone searches in parallel for maximum speed
    - Collects and organizes results by query/category
    - Handles fallback to internet search when needed
    - Provides comprehensive results for ToolSelectorAgent
    """

    def __init__(self):
        """Initialize the parallel search service."""
        self.pinecone_service = PineconeService()
        self.internet_service = InternetSearchService()
        self.recommender = AIToolRecommender()
        self._search_cache = {}  # Cache to prevent repeated searches

    async def execute_parallel_searches(
        self,
        search_queries: List[Dict[str, Any]],
        max_results_per_query: int = 5,
        similarity_threshold: float = 0.045,  # Proper threshold for Pinecone
    ) -> Dict[str, Any]:
        """
        Execute multiple tool searches in parallel.

        Args:
            search_queries: List of search query objects from QueryDecomposerAgent
            max_results_per_query: Maximum results per individual query
            similarity_threshold: Minimum similarity score for Pinecone results

        Returns:
            Dictionary containing:
            - search_results: Results organized by query
            - summary: Overall search summary
            - performance_metrics: Timing and performance data
            - needs_internet_search: List of queries that need internet fallback
        """
        try:
            logger.info(f"🚀 Starting parallel search for {len(search_queries)} queries")

            # Limit number of queries to prevent excessive searches
            if len(search_queries) > 10:
                logger.warning(
                    f"⚠️ Too many queries ({len(search_queries)}), limiting to 10"
                )
                search_queries = search_queries[:10]

            async with performance_monitor.time_operation("parallel_tool_search"):
                # Execute all Pinecone searches in parallel
                pinecone_results = await self._execute_pinecone_searches_parallel(
                    search_queries, max_results_per_query, similarity_threshold
                )

                # Analyze results and identify queries needing internet search
                needs_internet = self._identify_queries_needing_internet_search(
                    search_queries, pinecone_results
                )

                # Execute internet searches for queries with insufficient results
                internet_results = {}
                if needs_internet:
                    logger.info(
                        f"🌐 Executing internet search for {len(needs_internet)} queries"
                    )
                    internet_results = await self._execute_internet_searches_parallel(
                        needs_internet, max_results_per_query
                    )

                # Combine and organize all results
                combined_results = self._combine_search_results(
                    search_queries, pinecone_results, internet_results
                )

                # Generate summary and metrics
                summary = self._generate_search_summary(
                    combined_results, search_queries
                )

                return {
                    "search_results": combined_results,
                    "summary": summary,
                    "performance_metrics": {
                        "total_queries": len(search_queries),
                        "pinecone_queries": len(search_queries),
                        "internet_queries": len(needs_internet),
                        "total_tools_found": summary["total_tools_found"],
                        "search_method": "parallel_optimized",
                    },
                    "needs_internet_search": [q["query"] for q in needs_internet],
                    "status": "success",
                }

        except Exception as e:
            logger.error(f"❌ Error in parallel search execution: {e}", exc_info=True)
            return {
                "search_results": {},
                "summary": {"total_tools_found": 0, "successful_queries": 0},
                "performance_metrics": {},
                "needs_internet_search": [],
                "status": "error",
                "error": str(e),
            }

    async def _execute_pinecone_searches_parallel(
        self,
        search_queries: List[Dict[str, Any]],
        max_results: int,
        similarity_threshold: float,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute all Pinecone searches in parallel with caching."""
        try:
            logger.info(
                f"🔍 Executing {len(search_queries)} Pinecone searches in parallel"
            )

            # Check cache and deduplicate queries
            unique_queries = []
            results = {}

            for query_obj in search_queries:
                query_text = query_obj["query"]
                query_id = query_obj.get("target_category", f"query_{len(results)}")
                cache_key = (
                    f"pinecone:{query_text}:{max_results}:{similarity_threshold}"
                )

                # Check cache first
                if cache_key in self._search_cache:
                    logger.info(f"🚀 Cache hit for {query_id}")
                    results[query_id] = self._search_cache[cache_key]
                else:
                    unique_queries.append((query_obj, cache_key))

            # Only search for non-cached queries
            if unique_queries:
                logger.info(
                    f"🔍 Searching {len(unique_queries)} unique queries (cached: {len(search_queries) - len(unique_queries)})"
                )

                # Create tasks for parallel execution
                tasks = []
                for i, (query_obj, cache_key) in enumerate(unique_queries):
                    query_text = query_obj["query"]
                    task = asyncio.create_task(
                        self.pinecone_service.search_tools(
                            query=query_text,
                            max_results=max_results,
                            similarity_threshold=similarity_threshold,
                        ),
                        name=f"pinecone_search_{i}",
                    )
                    tasks.append((query_obj, cache_key, task))

                # Wait for all searches to complete
                completed_tasks = await asyncio.gather(
                    *[task for _, _, task in tasks], return_exceptions=True
                )

                # Process results and cache them
                for (query_obj, cache_key, _), result in zip(tasks, completed_tasks):
                    query_id = query_obj.get("target_category", f"query_{len(results)}")

                    if isinstance(result, Exception):
                        logger.error(
                            f"❌ Pinecone search failed for {query_id}: {result}"
                        )
                        search_result = []
                    else:
                        search_result = result or []
                        logger.info(
                            f"✅ Pinecone search for '{query_id}': {len(search_result)} tools"
                        )

                    # Cache the result
                    self._search_cache[cache_key] = search_result
                    results[query_id] = search_result

            total_tools = sum(len(tools) for tools in results.values())
            logger.info(
                f"🎯 Pinecone parallel search completed: {total_tools} total tools"
            )

            return results

        except Exception as e:
            logger.error(f"❌ Error in parallel Pinecone search: {e}")
            return {}

    def _identify_queries_needing_internet_search(
        self,
        search_queries: List[Dict[str, Any]],
        pinecone_results: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Identify which queries need internet search fallback."""
        needs_internet = []

        for query_obj in search_queries:
            query_id = query_obj.get("target_category", "unknown")
            expected_results = query_obj.get("expected_results", 3)
            priority = query_obj.get("priority", "medium")

            # Get actual results from Pinecone
            actual_results = len(pinecone_results.get(query_id, []))

            # Determine if internet search is needed
            needs_fallback = False

            if actual_results == 0:
                # No results at all - definitely need internet search
                needs_fallback = True
                logger.info(f"🌐 {query_id}: No Pinecone results, needs internet search")
            elif actual_results < expected_results and priority == "high":
                # High priority queries with insufficient results
                needs_fallback = True
                logger.info(
                    f"🌐 {query_id}: High priority with only {actual_results}/{expected_results} results"
                )
            elif actual_results < (expected_results // 2):
                # Less than half expected results
                needs_fallback = True
                logger.info(
                    f"🌐 {query_id}: Insufficient results {actual_results}/{expected_results}"
                )

            if needs_fallback:
                needs_internet.append(query_obj)

        logger.info(
            f"📊 {len(needs_internet)}/{len(search_queries)} queries need internet search"
        )
        return needs_internet

    async def _execute_internet_searches_parallel(
        self, queries_needing_internet: List[Dict[str, Any]], max_results: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute internet searches in parallel for queries that need them."""
        try:
            if not queries_needing_internet:
                return {}

            logger.info(
                f"🌐 Executing {len(queries_needing_internet)} internet searches in parallel"
            )

            # Create tasks for parallel internet searches
            tasks = []
            for query_obj in queries_needing_internet:
                query_text = query_obj["query"]
                task = asyncio.create_task(
                    self.internet_service.search_ai_tools(
                        query=query_text, max_results=max_results
                    ),
                    name=f"internet_search_{query_obj.get('target_category', 'unknown')}",
                )
                tasks.append((query_obj, task))

            # Wait for all internet searches to complete
            results = {}
            completed_tasks = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            # Process internet search results
            for (query_obj, _), result in zip(tasks, completed_tasks):
                query_id = query_obj.get(
                    "target_category", f"internet_query_{len(results)}"
                )

                if isinstance(result, Exception):
                    logger.error(f"❌ Internet search failed for {query_id}: {result}")
                    results[query_id] = []
                else:
                    results[query_id] = result or []
                    logger.info(
                        f"✅ Internet search for '{query_id}': {len(results[query_id])} tools"
                    )

            total_internet_tools = sum(len(tools) for tools in results.values())
            logger.info(
                f"🌐 Internet parallel search completed: {total_internet_tools} total tools"
            )

            return results

        except Exception as e:
            logger.error(f"❌ Error in parallel internet search: {e}")
            return {}

    def _combine_search_results(
        self,
        search_queries: List[Dict[str, Any]],
        pinecone_results: Dict[str, List[Dict[str, Any]]],
        internet_results: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """Combine Pinecone and internet search results by query."""
        combined_results = {}

        for query_obj in search_queries:
            query_id = query_obj.get(
                "target_category", f"query_{len(combined_results)}"
            )

            # Get results from both sources
            pinecone_tools = pinecone_results.get(query_id, [])
            internet_tools = internet_results.get(query_id, [])

            # Combine and deduplicate
            all_tools = pinecone_tools + internet_tools
            deduplicated_tools = self._deduplicate_tools(all_tools)

            # Sort by relevance (similarity score or relevance score)
            sorted_tools = sorted(
                deduplicated_tools,
                key=lambda x: x.get("Similarity Score", x.get("Relevance Score", 0)),
                reverse=True,
            )

            combined_results[query_id] = {
                "query_info": query_obj,
                "tools": sorted_tools,
                "pinecone_count": len(pinecone_tools),
                "internet_count": len(internet_tools),
                "total_count": len(sorted_tools),
                "sources": {
                    "pinecone": len(pinecone_tools) > 0,
                    "internet": len(internet_tools) > 0,
                },
            }

            logger.info(
                f"📊 {query_id}: {len(sorted_tools)} tools "
                f"(Pinecone: {len(pinecone_tools)}, Internet: {len(internet_tools)})"
            )

        return combined_results

    def _deduplicate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate tools using intelligent multi-factor deduplication."""
        seen_tools = set()
        deduplicated = []

        # Sort by quality first (keep highest quality duplicates)
        sorted_tools = sorted(
            tools,
            key=lambda x: x.get("Similarity Score", x.get("Relevance Score", 0)),
            reverse=True,
        )

        for tool in sorted_tools:
            # Create comprehensive identifiers
            identifiers = self._create_comprehensive_identifiers(tool)

            # Check if any identifier already exists
            is_duplicate = False
            matched_identifier = None
            for identifier in identifiers:
                if identifier in seen_tools:
                    is_duplicate = True
                    matched_identifier = identifier
                    break

            if not is_duplicate and identifiers:
                # Add all identifiers to seen set
                for identifier in identifiers:
                    seen_tools.add(identifier)
                deduplicated.append(tool)
            else:
                logger.debug(
                    f"🔄 Dedup: Skipping duplicate tool '{tool.get('Title', 'Unknown')}' (matched: {matched_identifier})"
                )

        logger.info(f"✅ Deduplication: {len(tools)} → {len(deduplicated)} tools")
        return deduplicated

    def _create_comprehensive_identifiers(self, tool: Dict[str, Any]) -> List[str]:
        """Create comprehensive identifiers for better deduplication."""
        identifiers = []

        try:
            title = tool.get("Title", "").strip()
            website = tool.get("Website", "").strip()

            # 1. Website-based identifiers (most reliable)
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

                    # Domain-only identifier
                    try:
                        from urllib.parse import urlparse

                        domain = urlparse(website).netloc.lower()
                        if domain:
                            identifiers.append(f"domain:{domain}")
                    except Exception:
                        pass

            # 2. Title-based identifiers
            if title and len(title) > 3:
                # Exact title
                identifiers.append(f"title:{title.lower()}")

                # Normalized title (handle variations)
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

                # Core name (remove common suffixes)
                core_name = (
                    title.lower()
                    .replace(" app", "")
                    .replace(" tool", "")
                    .replace(" platform", "")
                    .replace(" software", "")
                    .strip()
                )
                if len(core_name) > 3 and core_name != title.lower():
                    identifiers.append(f"core:{core_name}")

            # 3. Fallback identifier
            if not identifiers:
                identifiers.append(f"fallback:{title.lower()}" if title else "unknown")

            return identifiers

        except Exception as e:
            logger.error(f"Error creating comprehensive identifiers: {e}")
            return [f"error:{tool.get('Title', 'unknown')}"]

    def _generate_search_summary(
        self,
        combined_results: Dict[str, Dict[str, Any]],
        search_queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate summary of search results."""
        total_tools = sum(result["total_count"] for result in combined_results.values())

        successful_queries = len(
            [
                result
                for result in combined_results.values()
                if result["total_count"] > 0
            ]
        )

        categories_covered = list(combined_results.keys())

        # Calculate coverage by priority
        high_priority_queries = [
            q for q in search_queries if q.get("priority") == "high"
        ]
        high_priority_coverage = len(
            [
                q
                for q in high_priority_queries
                if combined_results.get(q.get("target_category", ""), {}).get(
                    "total_count", 0
                )
                > 0
            ]
        )

        return {
            "total_tools_found": total_tools,
            "successful_queries": successful_queries,
            "total_queries": len(search_queries),
            "categories_covered": categories_covered,
            "high_priority_coverage": f"{high_priority_coverage}/{len(high_priority_queries)}",
            "coverage_percentage": round(
                (successful_queries / len(search_queries)) * 100, 1
            ),
            "average_tools_per_query": round(total_tools / len(search_queries), 1)
            if search_queries
            else 0,
            "search_quality": "excellent"
            if successful_queries == len(search_queries)
            else "good"
            if successful_queries >= len(search_queries) * 0.7
            else "needs_improvement",
        }

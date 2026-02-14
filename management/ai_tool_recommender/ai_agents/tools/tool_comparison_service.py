"""Tool Comparison Service for Battle Cards functionality."""

import logging
import time
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.core.performance_monitor import performance_monitor
from ai_tool_recommender.ai_agents.core.redis_cache import redis_cache
from ai_tool_recommender.ai_agents.core.validation import ToolDataValidator
from ai_tool_recommender.ai_agents.tools.internet_search import InternetSearchService
from ai_tool_recommender.ai_agents.tools.pinecone import PineconeService

logger = logging.getLogger(__name__)


class ToolComparisonService:
    """Service for finding alternative tools for battle cards comparison."""

    def __init__(self):
        """Initialize the Tool Comparison service."""
        self.pinecone_service = PineconeService()
        self.internet_service = InternetSearchService()
        self._redis_connected = False
        logger.info("Tool Comparison service initialized")

    async def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            await redis_cache.connect()
            self._redis_connected = True
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self._redis_connected = False

    async def find_alternative_tools(
        self,
        original_tool_data: Dict[str, Any],
        node_id: str,
        max_results: int = 8,
        include_explanations: bool = True,
    ) -> Dict[str, Any]:
        """Find alternative tools for comparison with the original tool.

        Args:
            original_tool_data: Original tool data from workflow node
            node_id: Node ID in the workflow
            max_results: Maximum number of alternative tools to find
            include_explanations: Whether to include detailed explanations

        Returns:
            Dictionary with alternative tools and comparison data
        """
        try:
            start_time = time.time()
            logger.info(
                f"Finding alternatives for tool: {original_tool_data.get('label', 'Unknown')}"
            )

            # Generate comparison search query
            comparison_query = await self._generate_comparison_query(original_tool_data)
            logger.info(f"Generated comparison query: {comparison_query[:100]}...")

            # Check cache first
            cache_key = f"tool_comparison:{node_id}:{hash(comparison_query)}"
            if self._redis_connected:
                try:
                    cached_result = await redis_cache.get(cache_key)
                    if cached_result:
                        logger.info("🚀 Cache hit - returning cached comparison results")
                        return cached_result
                except Exception as e:
                    logger.warning(f"Cache retrieval failed: {e}")

            # Search for alternative tools in parallel
            async with performance_monitor.time_operation("comparison_search"):
                alternative_tools = await self._search_alternatives(
                    comparison_query, max_results
                )

            if not alternative_tools:
                return {
                    "status": "error",
                    "message": "No alternative tools found",
                    "alternatives": [],
                    "original_tool": original_tool_data,
                    "comparison_query": comparison_query,
                }

            # Filter out the original tool if it appears in results
            filtered_alternatives = await self._filter_original_tool(
                alternative_tools, original_tool_data
            )

            # Generate detailed comparisons if requested
            if include_explanations and filtered_alternatives:
                async with performance_monitor.time_operation("comparison_analysis"):
                    filtered_alternatives = await self._generate_tool_comparisons(
                        original_tool_data, filtered_alternatives
                    )

            # Validate alternative tools
            async with performance_monitor.time_operation("validation"):
                validator = ToolDataValidator()
                validated_alternatives = await validator.validate_tools_batch(
                    filtered_alternatives
                )

            total_time = time.time() - start_time

            result = {
                "status": "success",
                "original_tool": original_tool_data,
                "alternatives": validated_alternatives,
                "comparison_query": comparison_query,
                "total_found": len(validated_alternatives),
                "node_id": node_id,
                "performance": {
                    "total_time": round(total_time, 2),
                    "status": "fast" if total_time < 3.0 else "normal",
                },
            }

            # Cache the result
            if self._redis_connected:
                try:
                    await redis_cache.set(
                        cache_key, result, ttl=3600
                    )  # Cache for 1 hour
                except Exception as e:
                    logger.warning(f"Cache storage failed: {e}")

            return result

        except Exception as e:
            logger.error(f"Error in find_alternative_tools: {e}")
            import traceback

            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "alternatives": [],
                "original_tool": original_tool_data,
            }

    async def _generate_comparison_query(self, tool_data: Dict[str, Any]) -> str:
        """Generate a search query to find alternative tools."""
        try:
            # Extract key information from the tool
            tool_name = tool_data.get("label", "Unknown Tool")
            description = tool_data.get("description", "")
            features = tool_data.get("features", [])
            tags = tool_data.get("tags", [])

            # Create context for LLM
            tool_context = f"""
            Tool Name: {tool_name}
            Description: {description}
            Features: {', '.join(features) if features else 'Not specified'}
            Tags: {', '.join(tags) if tags else 'Not specified'}
            """

            prompt = f"""
            Given this AI tool information:
            {tool_context}

            Generate a search query to find alternative tools that serve similar purposes or solve similar problems.
            The query should focus on the core functionality and use case, not the specific tool name.

            Requirements:
            1. Focus on what the tool does, not what it's called
            2. Include key functionality keywords
            3. Make it broad enough to find alternatives but specific enough to be relevant
            4. Keep it under 100 characters
            5. Don't include the original tool name

            Return only the search query, no explanations.
            """

            response = await get_shared_llm().generate_response(prompt)
            query = response.strip().replace('"', "").replace("'", "")

            # Fallback query if LLM fails
            if not query or len(query) < 10:
                if features:
                    query = f"AI tools for {' '.join(features[:3])}"
                elif tags:
                    query = f"AI tools {' '.join(tags[:3])}"
                else:
                    query = f"AI tools similar to {tool_name}"

            logger.info(f"Generated comparison query: {query}")
            return query

        except Exception as e:
            logger.error(f"Error generating comparison query: {e}")
            # Fallback query
            tool_name = tool_data.get("label", "AI tool")
            return f"AI tools similar to {tool_name}"

    async def _search_alternatives(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Search for alternative tools using Pinecone first, then Internet if needed."""
        try:
            all_tools = []

            # Step 1: Search Pinecone first (fast)
            logger.info(f"🔍 Step 1: Searching Pinecone for alternatives...")
            pinecone_tools = await self.pinecone_service.search_tools(
                query, max_results
            )
            logger.info(f"✅ Pinecone returned {len(pinecone_tools)} alternative tools")
            all_tools.extend(pinecone_tools)

            # Step 2: If Pinecone has enough results, return immediately (fast path)
            if len(all_tools) >= max_results:
                logger.info(
                    f"✅ Found {len(all_tools)} tools from Pinecone - returning immediately (fast)"
                )
                return all_tools[:max_results]

            # Step 3: If not enough from Pinecone, search Internet (fallback)
            remaining = max_results - len(all_tools)
            if remaining > 0:
                logger.info(
                    f"🔍 Step 2: Only {len(all_tools)} tools from Pinecone, searching Internet for {remaining} more..."
                )
                try:
                    internet_tools = await self.internet_service.search_ai_tools(
                        query, remaining
                    )
                    logger.info(
                        f"✅ Internet returned {len(internet_tools)} alternative tools"
                    )
                    all_tools.extend(internet_tools)
                except Exception as e:
                    logger.warning(f"Internet search failed: {e}")
                    # Continue with Pinecone results only

            logger.info(f"✅ Total alternative tools found: {len(all_tools)}")
            return all_tools[:max_results]

        except Exception as e:
            logger.error(f"Error searching for alternatives: {e}")
            return []

    async def _filter_original_tool(
        self, alternatives: List[Dict[str, Any]], original_tool: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter out the original tool from alternatives if it appears."""
        original_name = original_tool.get("label", "").lower()
        original_website = original_tool.get("website", "").lower()

        filtered = []
        for tool in alternatives:
            tool_name = tool.get("Title", "").lower()
            tool_website = tool.get("Website", "").lower()

            # Skip if it's the same tool (by name or website)
            if (tool_name and original_name and tool_name == original_name) or (
                tool_website and original_website and tool_website == original_website
            ):
                logger.info(
                    f"Filtered out original tool: {tool.get('Title', 'Unknown')}"
                )
                continue

            filtered.append(tool)

        logger.info(f"Filtered alternatives: {len(filtered)} (removed original tool)")
        return filtered

    async def _generate_tool_comparisons(
        self, original_tool: Dict[str, Any], alternatives: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate detailed comparisons between original tool and alternatives."""
        try:
            # Limit to top alternatives for detailed analysis
            top_alternatives = alternatives[:6]

            for i, alternative in enumerate(top_alternatives):
                try:
                    comparison = await self._compare_tools(original_tool, alternative)
                    alternative["comparison"] = comparison
                    logger.info(f"Generated comparison {i + 1}/{len(top_alternatives)}")
                except Exception as e:
                    logger.error(f"Error comparing tool {i + 1}: {e}")
                    alternative["comparison"] = {
                        "summary": "Comparison unavailable",
                        "key_differences": [],
                        "pros": [],
                        "cons": [],
                    }

            # For remaining alternatives, add basic comparison
            for alternative in alternatives[6:]:
                alternative["comparison"] = {
                    "summary": f"Alternative to {original_tool.get('label', 'the original tool')}",
                    "key_differences": ["Detailed comparison not available"],
                    "pros": ["Alternative option available"],
                    "cons": [],
                }

            return alternatives

        except Exception as e:
            logger.error(f"Error generating tool comparisons: {e}")
            return alternatives

    async def _compare_tools(
        self, original_tool: Dict[str, Any], alternative_tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a detailed comparison between two tools."""
        try:
            original_name = original_tool.get("label", "Original Tool")
            alternative_name = alternative_tool.get("Title", "Alternative Tool")

            prompt = f"""
            Compare these two AI tools and provide a detailed analysis:

            **Original Tool: {original_name}**
            Description: {original_tool.get('description', '')}
            Features: {', '.join(original_tool.get('features', []))}
            Website: {original_tool.get('website', '')}

            **Alternative Tool: {alternative_name}**
            Description: {alternative_tool.get('Description', '')}
            Features: {alternative_tool.get('Features', '')}
            Website: {alternative_tool.get('Website', '')}

            Provide a comparison in this JSON format:
            {{
                "summary": "Brief 1-2 sentence comparison summary",
                "key_differences": ["Difference 1", "Difference 2", "Difference 3"],
                "pros": ["Pro 1", "Pro 2", "Pro 3"],
                "cons": ["Con 1", "Con 2"],
                "use_case_fit": "How well this alternative fits the original use case"
            }}

            Focus on practical differences that would matter to users choosing between these tools.
            Return only valid JSON, no explanations.
            """

            response = await get_shared_llm().generate_response(prompt)

            try:
                comparison_data = await get_shared_llm().parse_json_response(response)
                return comparison_data
            except Exception:
                # Fallback comparison
                return {
                    "summary": f"{alternative_name} is an alternative to {original_name}",
                    "key_differences": ["Different tool with similar functionality"],
                    "pros": ["Alternative option available"],
                    "cons": ["Detailed comparison not available"],
                    "use_case_fit": "May serve similar purposes",
                }

        except Exception as e:
            logger.error(f"Error in tool comparison: {e}")
            return {
                "summary": "Comparison unavailable",
                "key_differences": [],
                "pros": [],
                "cons": [],
                "use_case_fit": "Unknown",
            }

"""AI Tool Recommender service that combines Pinecone and Internet Search."""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.core.performance_monitor import performance_monitor
from ai_tool_recommender.ai_agents.core.redis_cache import query_cache, redis_cache
from ai_tool_recommender.ai_agents.core.validation import (
    ToolDataFormatter,
    ToolDataValidator,
)
from ai_tool_recommender.ai_agents.tools.internet_search import InternetSearchService
from ai_tool_recommender.ai_agents.tools.pinecone import PineconeService

logger = logging.getLogger(__name__)


class AIToolRecommender:
    """AI Tool Recommender service combining Pinecone and Internet Search."""

    def __init__(self):
        """Initialize the AI Tool Recommender service with Redis caching."""
        self.pinecone_service = PineconeService()
        self.internet_service = InternetSearchService()
        self._redis_connected = False
        self.workflow_prompt = self._create_workflow_prompt()
        logger.info("AI Tool Recommender service initialized with Redis caching")

    async def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            await redis_cache.connect()
            self._redis_connected = True
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self._redis_connected = False

    async def search_tools(
        self,
        query: str,
        max_results: int = 10,
        include_pinecone: bool = True,
        include_internet: bool = True,
        use_intelligent_search: bool = True,
        original_query: str = "",
    ) -> Dict[str, Any]:
        """Search for AI tools with intelligent decomposition or legacy method.

        Args:
            query: Search query (could be refined query)
            max_results: Maximum number of results
            include_pinecone: Whether to include Pinecone results
            include_internet: Whether to include internet search results
            use_intelligent_search: Whether to use the new intelligent pipeline
            original_query: Original user query for context

        Returns:
            Dictionary with search results and performance metrics
        """
        try:
            # Log with stack trace to identify where this is being called from
            import traceback

            # Determine search method
            if use_intelligent_search:
                logger.info(
                    f"🧠 [SEARCH_TOOLS] Starting INTELLIGENT search for: '{query}'"
                )
                logger.info(
                    f"Config: pinecone={include_pinecone}, internet={include_internet}, max={max_results}"
                )

                return await self._intelligent_search_pipeline(
                    query,
                    max_results,
                    include_pinecone,
                    include_internet,
                    original_query,
                )
            else:
                logger.info(f"🔍 Starting LEGACY tool search for: {query[:50]}...")
                return await self._legacy_search_pipeline(
                    query, max_results, include_pinecone, include_internet
                )

        except Exception as e:
            logger.error(f"Error in search_tools: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "message": str(e), "tools": []}

    async def _intelligent_search_pipeline(
        self,
        refined_query: str,
        max_results: int,
        include_pinecone: bool,
        include_internet: bool,
        original_query: str = "",
    ) -> Dict[str, Any]:
        """Execute the new intelligent search pipeline."""
        try:
            # Check cache first
            cache_key = f"intelligent_{refined_query[:100]}"
            if self._redis_connected:
                cached_results = await query_cache.get_query_results(
                    cache_key, max_results
                )
                if cached_results:
                    logger.info("🚀 Cache hit - returning cached intelligent results")
                    return {
                        **cached_results,
                        "performance": {
                            "total_time": 0.1,
                            "status": "cached",
                            "optimization_level": "intelligent_cached",
                        },
                    }

            logger.info("🧠 Executing intelligent search pipeline...")

            # Execute intelligent search without timeout - unlimited processing time per user request
            try:
                return await self._execute_intelligent_search_steps(
                    refined_query,
                    max_results,
                    include_pinecone,
                    include_internet,
                    original_query,
                )
            except Exception as e:
                logger.error(
                    f"❌ Error in intelligent search: {e}, falling back to legacy"
                )
                return await self._legacy_search_pipeline(
                    refined_query, max_results, include_pinecone, include_internet
                )
        except Exception as e:
            logger.error(f"❌ Error in intelligent search pipeline: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "tools": []}

    async def _execute_intelligent_search_steps(
        self,
        refined_query: str,
        max_results: int,
        include_pinecone: bool,
        include_internet: bool,
        original_query: str = "",
    ) -> Dict[str, Any]:
        """Execute the intelligent search steps with proper error handling."""
        try:
            # Step 1: Decompose refined query into targeted searches
            from ai_tool_recommender.agents.query_decomposer_agent import (
                QueryDecomposerAgent,
            )

            decomposer = QueryDecomposerAgent()
            decomposition = await decomposer.decompose_refined_query(
                refined_query, original_query
            )

            if not decomposition or not decomposition.get("search_queries"):
                logger.error(
                    "❌ Query decomposition failed, falling back to legacy search"
                )
                return await self._legacy_search_pipeline(
                    refined_query, max_results, include_pinecone, include_internet
                )

            search_queries = decomposition.get("search_queries", [])
            workflow_understanding = decomposition.get("workflow_understanding", "")

            logger.info(f"🎯 Decomposed into {len(search_queries)} targeted searches")

            # Step 2: Execute parallel searches
            from ai_tool_recommender.services.parallel_tool_search_service import (
                ParallelToolSearchService,
            )

            parallel_searcher = ParallelToolSearchService()
            search_results = await parallel_searcher.execute_parallel_searches(
                search_queries,
                max_results_per_query=5,  # 5 results per query
                similarity_threshold=0.045,  # Proper threshold for Pinecone
            )

            if search_results.get("status") == "error":
                logger.error("❌ Parallel search failed, falling back to legacy search")
                return await self._legacy_search_pipeline(
                    refined_query, max_results, include_pinecone, include_internet
                )

            total_found = search_results.get("summary", {}).get("total_tools_found", 0)
            logger.info(
                f"🔍 Parallel search found {total_found} tools across all categories"
            )

            # Step 3: Intelligent tool selection
            from ai_tool_recommender.agents.tool_selector_agent import ToolSelectorAgent

            selector = ToolSelectorAgent()
            selection_result = await selector.select_optimal_tools(
                search_results.get("search_results", {}),
                workflow_understanding,
                refined_query,
                max_tools=max_results,
            )

            if selection_result.get("status") == "error":
                logger.error("❌ Tool selection failed, falling back to legacy search")
                return await self._legacy_search_pipeline(
                    refined_query, max_results, include_pinecone, include_internet
                )

            selected_tools = selection_result.get("selected_tools", [])
            logger.info(f"🎯 Selected {len(selected_tools)} optimal tools")

            # Step 4: Validate selected tools
            async with performance_monitor.time_operation("validation"):
                validator = ToolDataValidator()
                validated_tools = await validator.validate_tools_batch(selected_tools)
                logger.info(f"✅ Validated {len(validated_tools)} tools")

            # Calculate performance metrics
            import time

            start_time = time.time()
            total_time = time.time() - start_time
            performance_monitor.record_request(total_time)

            # Count tools by source
            pinecone_count = len(
                [
                    t
                    for t in validated_tools
                    if "Pinecone" in t.get("Source", "")
                    or "Scraped" in t.get("Source", "")
                ]
            )
            internet_count = len(
                [t for t in validated_tools if "Internet Search" in t.get("Source", "")]
            )

            # 💾 Save Internet Search tools to PostgreSQL for immediate tool chat access
            if internet_count > 0:
                try:
                    saved_count = await self._save_internet_tools_to_database(
                        validated_tools
                    )
                    logger.info(
                        f"💾 Saved {saved_count} Internet Search tools to database for tool chat"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error saving Internet Search tools to database: {e}"
                    )

            # Prepare comprehensive response
            response = {
                "status": "success",
                "tools": validated_tools,
                "message": f"Found {len(validated_tools)} optimal tools using intelligent search",
                "total_count": len(validated_tools),
                "count": len(validated_tools),  # For backward compatibility
                "pinecone_count": pinecone_count,
                "internet_count": internet_count,
                "search_method": "intelligent_pipeline",
                "decomposition": {
                    "workflow_understanding": workflow_understanding,
                    "search_queries_used": len(search_queries),
                    "categories_searched": list(
                        search_results.get("search_results", {}).keys()
                    ),
                },
                "selection_analysis": {
                    "selection_reasoning": selection_result.get(
                        "selection_reasoning", ""
                    ),
                    "workflow_coverage": selection_result.get("workflow_coverage", {}),
                    "confidence_score": selection_result.get("confidence_score", 0),
                },
                "performance": {
                    "total_time": round(total_time, 2),
                    "status": "intelligent" if total_time < 5.0 else "normal",
                    "optimization_level": "intelligent_pipeline",
                    "parallel_searches": len(search_queries),
                },
            }

            # Cache results
            if self._redis_connected:
                cache_key = f"intelligent_{refined_query[:100]}"
                await query_cache.set_query_results(cache_key, max_results, response)

            return response

        except Exception as e:
            logger.error(f"❌ Error in intelligent search steps: {e}", exc_info=True)
            raise  # Re-raise to be caught by timeout handler

    async def _legacy_search_pipeline(
        self,
        query: str,
        max_results: int,
        include_pinecone: bool,
        include_internet: bool,
    ) -> Dict[str, Any]:
        """Execute the original search pipeline as fallback."""
        try:
            import time

            start_time = time.time()

            logger.info(f"🔍 Executing legacy search pipeline for: {query[:50]}...")

            # CRITICAL SAFETY: Never search Pinecone/Internet with a massive query block.
            # If query is > 500 chars, it's likely a Refined Query block being misused.
            if len(query) > 500:
                logger.warning(
                    f"⚠️ Query is too long ({len(query)} chars), truncating for legacy pipeline safety."
                )
                # Strategy: Take first 300 chars or the first descriptive line.
                query = query[:300].split("\n")[0]
                if (
                    len(query) < 50
                ):  # If first line was too short, just take first 300 chars.
                    query = query[:300]
                logger.info(f"🎯 Safely truncated legacy query: '{query}'")

            # Check cache first
            if self._redis_connected:
                cached_results = await query_cache.get_query_results(query, max_results)
                if cached_results:
                    logger.info("🚀 Cache hit - returning cached results")
                    return {
                        **cached_results,
                        "performance": {
                            "total_time": 0.1,
                            "status": "cached",
                            "optimization_level": "legacy_cached",
                        },
                    }
            else:
                # Try to initialize Redis if not connected
                await self._initialize_redis()

            # 🎯 HYBRID FLOW: Validate Pinecone tools, supplement with Internet search
            async with performance_monitor.time_operation("total_search"):
                all_tools = []
                pinecone_tools = []
                validated_pinecone_tools = []
                internet_tools = []

                # Step 1: Search Pinecone FIRST
                if include_pinecone:
                    print("=" * 100)
                    print("🔍🔍🔍 PINECONE SEARCH STARTING 🔍🔍🔍")
                    print(f"Query: '{query}'")
                    print(f"Max results: {max_results}")
                    print("=" * 100)

                    logger.info("=" * 80)
                    logger.info(
                        "🔍 [PINECONE HYBRID] ===== STARTING PINECONE SEARCH ====="
                    )
                    logger.info(f"📝 [PINECONE HYBRID] Query: '{query}'")
                    logger.info(f"📈 [PINECONE HYBRID] Max results: {max_results}")
                    logger.info("=" * 80)

                    # Create enhanced search query focused on tool descriptions
                    enhanced_query = await self._create_enhanced_search_query(query)
                    logger.info(f"🎯 [PINECONE HYBRID] Enhanced query: {enhanced_query}")

                    # Get more tools from Pinecone for validation (2x max_results)
                    pinecone_tools = await self.pinecone_service.search_tools(
                        enhanced_query,
                        max_results * 2,  # Get more candidates for validation
                    )

                    print("=" * 100)
                    print(
                        f"✅ PINECONE SEARCH COMPLETED: {len(pinecone_tools)} tools found"
                    )
                    print("=" * 100)

                    logger.info("=" * 80)
                    logger.info(
                        f"✅ [PINECONE HYBRID] Pinecone returned {len(pinecone_tools)} tools"
                    )
                    logger.info("=" * 80)

                    # Step 1a: Deduplicate Pinecone tools
                    if pinecone_tools:
                        logger.info(f"📦 [PINECONE RESULTS] Tools from Pinecone:")
                        for idx, tool in enumerate(
                            pinecone_tools[:10], 1
                        ):  # Log first 10
                            logger.info(
                                f"  {idx}. {tool.get('Title', 'Unknown')} - Source: {tool.get('Source', 'Unknown')}"
                            )

                        logger.info("🔍 [DEDUPLICATION] Checking for duplicate tools...")
                        deduplicated_tools = []
                        seen_identifiers = set()

                        for tool in pinecone_tools:
                            identifiers = self._create_tool_identifiers(tool)
                            is_duplicate = False
                            for identifier in identifiers:
                                if identifier in seen_identifiers:
                                    is_duplicate = True
                                    logger.warning(
                                        f"🔄 [DUPLICATE] Skipped duplicate: {tool.get('Title', 'Unknown')} ({identifier})"
                                    )
                                    break

                            if not is_duplicate:
                                seen_identifiers.update(identifiers)
                                deduplicated_tools.append(tool)

                        logger.info(
                            f"✅ [DEDUPLICATION] Removed {len(pinecone_tools) - len(deduplicated_tools)} duplicates"
                        )
                        logger.info(
                            f"✅ [DEDUPLICATION] Final count: {len(deduplicated_tools)} unique tools"
                        )

                        # Step 1b: VALIDATE Pinecone tools with LLM
                        logger.info("=" * 80)
                        logger.info(
                            f"🤖 [LLM VALIDATION] Validating {len(deduplicated_tools)} Pinecone tools for relevance..."
                        )
                        logger.info("=" * 80)

                        validated_pinecone_tools = await self._validate_tools_with_llm(
                            deduplicated_tools, query, max_results // 2
                        )

                        logger.info("=" * 80)
                        logger.info(
                            f"✅ [LLM VALIDATION] {len(validated_pinecone_tools)} tools passed validation"
                        )
                        logger.info(
                            f"❌ [LLM VALIDATION] {len(deduplicated_tools) - len(validated_pinecone_tools)} tools rejected as irrelevant"
                        )
                        logger.info("=" * 80)

                        if validated_pinecone_tools:
                            logger.info(
                                f"📦 [VALIDATED PINECONE] Good tools from Pinecone:"
                            )
                            for idx, tool in enumerate(validated_pinecone_tools, 1):
                                logger.info(
                                    f"  {idx}. {tool.get('Title', 'Unknown')} - {tool.get('_validation_reason', 'Relevant')}"
                                )

                        all_tools.extend(validated_pinecone_tools)

                    else:
                        logger.warning("⚠️ [PINECONE EMPTY] Pinecone returned 0 tools")

                # Step 2: Supplement with Internet search if needed
                tools_needed = max_results - len(all_tools)

                if include_internet and tools_needed > 0:
                    print("=" * 100)
                    print(
                        f"🌐🌐🌐 INTERNET SEARCH SUPPLEMENT (Need {tools_needed} more tools) 🌐🌐🌐"
                    )
                    print(f"Query: '{query}'")
                    print(f"Pinecone tools: {len(all_tools)}, Need: {tools_needed}")
                    print("=" * 100)

                    logger.info("=" * 80)
                    logger.info(
                        "🌐 [INTERNET SUPPLEMENT] ===== STARTING INTERNET SEARCH ====="
                    )
                    logger.info(
                        f"📊 [INTERNET SUPPLEMENT] Pinecone validated: {len(all_tools)} tools"
                    )
                    logger.info(f"🔍 [INTERNET SUPPLEMENT] Query: '{query}'")
                    logger.info(
                        f"📈 [INTERNET SUPPLEMENT] Need {tools_needed} more tools"
                    )
                    logger.info("=" * 80)

                    try:
                        logger.info(
                            f"🚀 [INTERNET SUPPLEMENT] Calling InternetSearchService.search_ai_tools()..."
                        )
                        logger.info(
                            f"⏱️ [INTERNET SUPPLEMENT] Starting at: {__import__('datetime').datetime.now().isoformat()}"
                        )

                        internet_tools = await self.internet_service.search_ai_tools(
                            query, tools_needed
                        )

                        logger.info(
                            f"⏱️ [INTERNET SUPPLEMENT] Completed at: {__import__('datetime').datetime.now().isoformat()}"
                        )
                        logger.info("=" * 80)
                        logger.info(
                            f"✅ [INTERNET RESULT] Internet search returned {len(internet_tools)} tools"
                        )
                        logger.info("=" * 80)

                        if internet_tools:
                            print("=" * 100)
                            print(
                                f"✅✅✅ INTERNET SEARCH FOUND {len(internet_tools)} TOOLS ✅✅✅"
                            )
                            print("=" * 100)

                            logger.info(
                                f"📦 [INTERNET RESULT] ═══════════════════════════════════════════════════"
                            )
                            logger.info(
                                f"📦 [INTERNET RESULT] TOOLS FROM INTERNET SEARCH:"
                            )
                            logger.info(
                                f"📦 [INTERNET RESULT] ═══════════════════════════════════════════════════"
                            )
                            for idx, tool in enumerate(internet_tools, 1):
                                print(
                                    f"  {idx}. {tool.get('Title', 'Unknown')} - {tool.get('Website', 'No URL')}"
                                )
                                logger.info(
                                    f"  ┌─ Tool #{idx} FROM INTERNET ─────────────────────────────────────────────"
                                )
                                logger.info(
                                    f"  │ Title: {tool.get('Title', 'Unknown')}"
                                )
                                logger.info(
                                    f"  │ Website: {tool.get('Website', 'No URL')}"
                                )
                                logger.info(
                                    f"  │ Description: {tool.get('Description', 'N/A')[:100]}..."
                                )
                                logger.info(
                                    f"  │ Source: {tool.get('Source', 'Unknown')}"
                                )
                                logger.info(
                                    f"  │ Category: {tool.get('Category', 'N/A')}"
                                )
                                logger.info(
                                    f"  └─────────────────────────────────────────────────────────"
                                )

                            print("=" * 100)

                            logger.info(
                                f"✅ [INTERNET RESULT] Adding {len(internet_tools)} Internet tools to results"
                            )
                            all_tools.extend(internet_tools)
                            logger.info(
                                f"📊 [HYBRID RESULT] Total tools: {len(all_tools)} (Pinecone: {len(validated_pinecone_tools)}, Internet: {len(internet_tools)})"
                            )

                            # Mark internet tools for background scraping
                            logger.info(
                                f"📝 [INTERNET RESULT] {len(internet_tools)} internet tools will be scraped in background"
                            )
                        else:
                            logger.warning(
                                "⚠️ [INTERNET RESULT] Internet search returned 0 tools - no results found"
                            )
                            logger.warning("⚠️ [INTERNET RESULT] This could mean:")
                            logger.warning("   1. No tools match the query")
                            logger.warning("   2. All results were filtered out")
                            logger.warning("   3. Search API error occurred")

                        logger.info("=" * 80)
                        logger.info(
                            "🌐 [INTERNET SUPPLEMENT] ===== INTERNET SEARCH COMPLETED ====="
                        )
                        logger.info("=" * 80)

                    except Exception as e:
                        print("=" * 100)
                        print(f"❌❌❌ INTERNET SEARCH ERROR: {e}")
                        print(f"Error type: {type(e).__name__}")
                        import traceback

                        print(f"Traceback:\n{traceback.format_exc()}")
                        print("=" * 100)

                        logger.error("=" * 80)
                        logger.error(
                            "❌ [INTERNET ERROR] ===== INTERNET SEARCH FAILED ====="
                        )
                        logger.error(f"❌ [INTERNET ERROR] Error: {str(e)}")
                        logger.error(
                            f"❌ [INTERNET ERROR] Error type: {type(e).__name__}"
                        )
                        logger.error(
                            f"❌ [INTERNET ERROR] Traceback:\n{traceback.format_exc()}"
                        )
                        logger.error("=" * 80)
                        # Continue with Pinecone results only
                elif include_internet and tools_needed <= 0:
                    print("=" * 100)
                    print(
                        f"⏭️⏭️⏭️ INTERNET SEARCH SKIPPED - Already have {len(all_tools)} tools ⏭️⏭️⏭️"
                    )
                    print(f"Validated Pinecone tools: {len(all_tools)}")
                    print("=" * 100)

                    logger.info("=" * 80)
                    logger.info(
                        f"⏭️ [SKIP INTERNET] Internet search SKIPPED - Already have {len(all_tools)} tools"
                    )
                    logger.info(
                        "✅ [SKIP INTERNET] Returning Pinecone tools without Internet search"
                    )
                    logger.info("=" * 80)
                else:
                    print("=" * 100)
                    print(
                        "⚠️⚠️⚠️ INTERNET SEARCH DISABLED - include_internet is FALSE ⚠️⚠️⚠️"
                    )
                    print(f"include_internet = {include_internet}")
                    print("=" * 100)

                    logger.warning("=" * 80)
                    logger.warning(
                        "⚠️ [INTERNET DISABLED] Internet search was DISABLED"
                    )
                    logger.warning(
                        f"⚠️ [INTERNET DISABLED] include_internet = {include_internet}"
                    )
                    logger.warning("=" * 80)
                    logger.info(
                        f"ℹ️ [INTERNET DISABLED] include_internet=False, skipping Internet search"
                    )
                    logger.info(f"✅ Found {len(all_tools)} tools from Pinecone only")

            # Final summary: Show which tools came from which source
            pinecone_count = len(
                [
                    t
                    for t in all_tools
                    if "Pinecone" in t.get("Source", "")
                    or "Scraped" in t.get("Source", "")
                    or "PostgreSQL" in t.get("Source", "")
                ]
            )
            internet_count = len(
                [t for t in all_tools if "Internet Search" in t.get("Source", "")]
            )

            print("=" * 100)
            print("📊📊📊 FINAL SEARCH SUMMARY 📊📊📊")
            print(f"Total tools found: {len(all_tools)}")
            print(f"  - From Pinecone: {pinecone_count}")
            print(f"  - From Internet Search: {internet_count}")
            print("=" * 100)

            logger.info("=" * 80)
            logger.info("📊 [FINAL SUMMARY] ===== SEARCH RESULTS SUMMARY =====")
            logger.info(f"📊 [FINAL SUMMARY] Total tools: {len(all_tools)}")
            logger.info(f"📊 [FINAL SUMMARY]   - From Pinecone: {pinecone_count}")
            logger.info(f"📊 [FINAL SUMMARY]   - From Internet Search: {internet_count}")
            logger.info("=" * 80)

            # List all Internet tools clearly
            if internet_count > 0:
                logger.info("=" * 80)
                logger.info("🌐 [INTERNET TOOLS] Tools discovered via Internet Search:")
                logger.info("=" * 80)
                internet_tools_list = [
                    t for t in all_tools if "Internet Search" in t.get("Source", "")
                ]
                for idx, tool in enumerate(internet_tools_list, 1):
                    logger.info(
                        f"  {idx}. {tool.get('Title', 'Unknown')} - {tool.get('Website', 'No URL')}"
                    )
                logger.info("=" * 80)

            if not all_tools:
                logger.warning("No tools found from any source")
                return {
                    "status": "error",
                    "message": "No relevant AI tools found",
                    "tools": [],
                }

            # 🎯 HYBRID APPROACH: Return validated Pinecone + Internet tools together
            logger.info("=" * 80)
            logger.info(
                f"✅ [HYBRID SUCCESS] Returning {len(all_tools)} tools (Pinecone: {pinecone_count}, Internet: {internet_count})"
            )
            logger.info("=" * 80)

            print("=" * 100)
            print(f"✅✅✅ RETURNING {len(all_tools)} HYBRID TOOLS ✅✅✅")
            print(f"  - Validated Pinecone: {pinecone_count}")
            print(f"  - Internet Search: {internet_count}")
            print("=" * 100)

            # Limit to max_results (should already be limited, but double-check)
            validated_tools = all_tools[:max_results]

            # Calculate performance metrics
            total_time = time.time() - start_time
            performance_monitor.record_request(total_time)

            # Prepare response
            response = {
                "status": "success",
                "tools": validated_tools,
                "message": f"Found {len(validated_tools)} relevant tools ({pinecone_count} from Pinecone, {internet_count} from Internet)",
                "total_count": len(validated_tools),
                "count": len(validated_tools),
                "pinecone_count": pinecone_count,
                "internet_count": internet_count,
                "search_method": "hybrid_search",
                "performance": {
                    "total_time": round(total_time, 2),
                    "status": "fast",
                    "optimization_level": "hybrid_validated",
                },
            }

            # Cache results for future requests
            if self._redis_connected:
                await query_cache.set_query_results(query, max_results, response)

            logger.info("=" * 80)
            logger.info(
                f"✅ [SUCCESS] Returning {len(validated_tools)} tools in {total_time:.2f}s (Pinecone: {pinecone_count}, Internet: {internet_count})"
            )
            logger.info("=" * 80)

            return response

            # OLD LEGACY CODE BELOW - NOT EXECUTED

            # If we're here, we only have Internet search results (Pinecone returned 0)
            # Apply filtering and validation only to Internet results
            logger.info("=" * 80)
            logger.info(
                "🌐 [INTERNET ONLY] Only Internet search results - will filter and validate"
            )
            logger.info("=" * 80)

            # Store internet tools count for background processing
            # (will be used by views to trigger scraping)
            if internet_tools:
                # Add metadata to response so views can identify internet tools
                for tool in all_tools:
                    if tool in internet_tools:
                        tool["_is_internet_tool"] = True

            logger.info(f"Found {len(all_tools)} tools from all sources")

            # Fast filtering
            async with performance_monitor.time_operation("filtering"):
                filtered_tools = await self._filter_tools(query, all_tools)

            if not filtered_tools:
                logger.warning("No tools selected after filtering")
                return {
                    "status": "error",
                    "message": "No tools selected after filtering",
                    "tools": [],
                }

            logger.info(f"Selected {len(filtered_tools)} tools after filtering")

            # Fast validation with parallel processing - SKIP FOR GEMINI TOOLS
            async with performance_monitor.time_operation("validation"):
                # Separate Gemini tools from others
                gemini_filtered = [
                    t
                    for t in filtered_tools
                    if "Internet Search" in t.get("Source", "")
                ]
                other_filtered = [
                    t
                    for t in filtered_tools
                    if "Internet Search" not in t.get("Source", "")
                ]

                logger.info(
                    f"🌐 Skipping validation for {len(gemini_filtered)} Gemini tools (accept all)"
                )
                logger.info(f"📦 Validating {len(other_filtered)} non-Gemini tools")

                # Validate only non-Gemini tools
                validator = ToolDataValidator()
                validated_other_tools = await validator.validate_tools_batch(
                    other_filtered
                )

                # Combine: ALL Gemini tools + validated others
                validated_tools = gemini_filtered + validated_other_tools

                logger.info(
                    f"Validated {len(validated_tools)} tools total ({len(gemini_filtered)} Gemini + {len(validated_other_tools)} others)"
                )

            # Calculate performance metrics
            total_time = time.time() - start_time
            performance_monitor.record_request(total_time)

            # Count tools by source
            pinecone_count = len(
                [
                    t
                    for t in validated_tools
                    if "Internet Search" not in t.get("Source", "")
                ]
            )
            internet_count = len(
                [t for t in validated_tools if "Internet Search" in t.get("Source", "")]
            )

            # Print final summary after validation
            print("=" * 100)
            print("📊📊📊 FINAL VALIDATED TOOLS SUMMARY 📊📊📊")
            print(f"Total validated tools: {len(validated_tools)}")
            print(f"  - From Pinecone: {pinecone_count}")
            print(f"  - From Gemini Internet Search: {internet_count}")
            if internet_count > 0:
                print("\n🌐 Tools from Gemini Internet Search:")
                gemini_tools = [
                    t
                    for t in validated_tools
                    if "Internet Search" in t.get("Source", "")
                ]
                for idx, tool in enumerate(gemini_tools, 1):
                    print(
                        f"  {idx}. {tool.get('Title', 'Unknown')} - {tool.get('Website', 'No URL')}"
                    )
            print("=" * 100)

            logger.info("=" * 80)
            logger.info("📊 [FINAL SUMMARY] ===== VALIDATED TOOLS SUMMARY =====")
            logger.info(
                f"📊 [FINAL SUMMARY] Total validated tools: {len(validated_tools)}"
            )
            logger.info(f"📊 [FINAL SUMMARY]   - From Pinecone: {pinecone_count}")
            logger.info(
                f"📊 [FINAL SUMMARY]   - From Gemini Internet Search: {internet_count}"
            )
            logger.info("=" * 80)

            # List all Gemini tools clearly
            if internet_count > 0:
                logger.info("=" * 80)
                logger.info(
                    "🌐 [GEMINI TOOLS] Tools discovered via Gemini Internet Search:"
                )
                logger.info("=" * 80)
                gemini_tools_list = [
                    t
                    for t in validated_tools
                    if "Internet Search" in t.get("Source", "")
                ]
                for idx, tool in enumerate(gemini_tools_list, 1):
                    logger.info(
                        f"  {idx}. {tool.get('Title', 'Unknown')} - {tool.get('Website', 'No URL')}"
                    )
                logger.info("=" * 80)

            # 💾 Save Internet Search tools to PostgreSQL for immediate tool chat access
            if internet_count > 0:
                try:
                    saved_count = await self._save_internet_tools_to_database(
                        validated_tools
                    )
                    logger.info(
                        f"💾 Saved {saved_count} Internet Search tools to database for tool chat"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Error saving Internet Search tools to database: {e}"
                    )

            # Prepare response
            response = {
                "status": "success",
                "tools": validated_tools,
                "message": f"Found {len(validated_tools)} relevant tools with complete data",
                "total_count": len(validated_tools),
                "count": len(validated_tools),  # For backward compatibility
                "pinecone_count": pinecone_count,
                "internet_count": internet_count,
                "search_method": "legacy_pipeline",
                "validation_report": validator.get_validation_report(validated_tools),
                "performance": {
                    "total_time": round(total_time, 2),
                    "status": performance_monitor.get_performance_report()[
                        "performance_status"
                    ],
                    "optimization_level": "legacy_performance",
                },
            }

            # Cache results for future requests
            if self._redis_connected:
                await query_cache.set_query_results(query, max_results, response)

            return response

        except Exception as e:
            logger.error(f"Error in legacy search pipeline: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "message": str(e), "tools": []}

    async def _create_enhanced_search_query(self, original_query: str) -> str:
        """
        Create an enhanced search query focused on tool descriptions and functionality.

        Args:
            original_query: Original user query

        Returns:
            Enhanced search query for better Pinecone results
        """
        try:
            # Use LLM to extract key functionality and create focused search terms
            prompt = f"""Transform this user query into a focused search query for finding AI tools.

Original Query: "{original_query}"

Your task:
1. Extract the main functionality/capability the user needs
2. Identify key action words and domains
3. Create a concise search query focused on tool descriptions
4. Include relevant synonyms and related terms

Rules:
- Focus on WHAT the tool should DO, not just tool names
- Use action words (automate, manage, create, analyze, etc.)
- Include domain-specific terms
- Keep it concise (max 10-15 words)
- Don't include "tool" or "software" - focus on functionality

Examples:
Input: "I need tools for payroll management"
Output: "payroll processing employee compensation salary benefits administration"

Input: "Help me automate social media posting"
Output: "social media automation content scheduling posting publishing"

Input: "I want to create workflows for customer support"
Output: "customer support workflow automation ticket management help desk"

Now transform the query above. Return ONLY the enhanced search query, no explanations."""

            enhanced_query = await get_shared_llm().generate_response(prompt)

            # Clean up the response
            enhanced_query = enhanced_query.strip().strip('"').strip("'")

            # Fallback if LLM fails
            if not enhanced_query or len(enhanced_query) < 5:
                logger.warning("LLM failed to enhance query, using original")
                return original_query

            logger.info(f"✅ Enhanced query: '{original_query}' → '{enhanced_query}'")
            return enhanced_query

        except Exception as e:
            logger.error(f"Error creating enhanced search query: {e}")
            return original_query  # Fallback to original query

    def _create_tool_identifiers(self, tool: Dict[str, Any]) -> List[str]:
        """Create multiple identifiers for intelligent deduplication.

        Args:
            tool: Tool dictionary

        Returns:
            List of unique identifiers for this tool
        """
        identifiers = []

        try:
            # Get tool data
            title = tool.get("Title", "").strip()
            website = tool.get("Website", "").strip()

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
            return [f"error:{tool.get('Title', 'unknown')}"]

    async def _validate_tools_with_llm(
        self, tools: List[Dict[str, Any]], query: str, max_tools: int
    ) -> List[Dict[str, Any]]:
        """Validate tools using LLM to check relevance to user's query.

        Args:
            tools: List of tools from Pinecone
            query: User's search query
            max_tools: Maximum number of tools to return

        Returns:
            List of validated tools that are relevant to the query
        """
        if not tools:
            return []

        try:
            # LLM is already imported at top of file
            # from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

            logger.info(
                f"🤖 [LLM VALIDATION] Validating {len(tools)} tools for query: '{query}'"
            )

            # Create a prompt for the LLM to evaluate each tool with RELEVANCY SCORES
            tools_summary = []
            for idx, tool in enumerate(tools[:20], 1):  # Limit to 20 for LLM context
                tools_summary.append(
                    f"{idx}. {tool.get('Title', 'Unknown')}: {tool.get('Description', 'No description')[:150]}"
                )

            prompt = f"""You are evaluating AI tools for relevance to a user's workflow query.

User's Query: "{query}"

Tools to evaluate:
{chr(10).join(tools_summary)}

Task: Rate each tool's relevance on a scale of 0-10:
- 10 = Perfect match, directly solves the user's need
- 7-9 = Highly relevant, very useful for this workflow
- 4-6 = Somewhat relevant, could be helpful
- 1-3 = Tangentially related, minor connection
- 0 = Completely irrelevant, no connection

Be LENIENT: If a tool has ANY reasonable connection to the query, give it at least a 4.
Only give 0-3 for tools that are completely unrelated.

Return your ratings in this EXACT format (one per line):
1:8
2:5
3:0
4:7
...

Your response:"""

            llm_response = await get_shared_llm().generate_response(prompt)
            llm_response = llm_response.strip()

            logger.info(f"🤖 [LLM VALIDATION] LLM response:\n{llm_response}")

            # Parse the LLM response to get tool scores
            validated_tools = []
            try:
                for line in llm_response.split("\n"):
                    line = line.strip()
                    if not line or ":" not in line:
                        continue

                    try:
                        idx_str, score_str = line.split(":")
                        idx = int(idx_str.strip()) - 1  # Convert to 0-indexed
                        score = int(score_str.strip())

                        # Accept tools with score >= 4 (somewhat relevant or better)
                        if 0 <= idx < len(tools) and score >= 4:
                            tool = tools[idx].copy()
                            tool["_validation_reason"] = f"Relevancy score: {score}/10"
                            tool["_validation_score"] = score / 10.0
                            validated_tools.append(tool)

                            logger.info(
                                f"  ✅ Tool #{idx+1} '{tool.get('Title', 'Unknown')}' - Score: {score}/10 - ACCEPTED"
                            )
                        elif 0 <= idx < len(tools):
                            logger.info(
                                f"  ❌ Tool #{idx+1} '{tools[idx].get('Title', 'Unknown')}' - Score: {score}/10 - REJECTED (too low)"
                            )

                    except (ValueError, IndexError) as e:
                        logger.warning(f"  ⚠️ Could not parse line: '{line}' - {e}")
                        continue

                # Sort by validation score (highest first) and take top max_tools
                validated_tools.sort(
                    key=lambda x: x.get("_validation_score", 0), reverse=True
                )
                validated_tools = validated_tools[:max_tools]

                logger.info("=" * 80)
                logger.info(
                    f"✅ [LLM VALIDATION] Accepted {len(validated_tools)} tools (score >= 4/10)"
                )
                logger.info(
                    f"❌ [LLM VALIDATION] Rejected {len(tools) - len(validated_tools)} tools (score < 4/10)"
                )
                logger.info("=" * 80)

                return validated_tools

            except Exception as parse_error:
                logger.error(
                    f"❌ [LLM VALIDATION] Error parsing LLM response: {parse_error}"
                )
                # Fallback: return top tools by similarity score
                logger.info(
                    "⚠️ [LLM VALIDATION] Falling back to similarity-based selection"
                )
                return tools[:max_tools]

        except Exception as e:
            logger.error(f"❌ [LLM VALIDATION] Error in LLM validation: {e}")
            # Fallback: return top tools by similarity score
            logger.info("⚠️ [LLM VALIDATION] Error occurred, falling back to top tools")
            return tools[:max_tools]

    async def _search_all_sources(
        self,
        query: str,
        max_results: int,
        include_pinecone: bool,
        include_internet: bool,
    ) -> List[Dict[str, Any]]:
        """Search all available sources for AI tools in parallel for maximum speed."""
        import asyncio

        all_tools = []
        tasks = []

        # Create parallel tasks for both sources
        if include_pinecone:
            pinecone_task = asyncio.create_task(
                self.pinecone_service.search_tools(query, max_results // 2)
            )
            tasks.append(("pinecone", pinecone_task))

        if include_internet:
            logger.info(
                f"🌐 [GEMINI TRIGGER] Starting parallel Gemini web search for query: '{query}'"
            )
            internet_task = asyncio.create_task(
                self.internet_service.search_ai_tools(query, max_results // 2)
            )
            tasks.append(("internet", internet_task))

        # Wait for all tasks to complete in parallel
        if tasks:
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            for _i, (source, result) in enumerate(
                zip([name for name, _ in tasks], results)
            ):
                if isinstance(result, Exception):
                    logger.error(f"❌ [GEMINI ERROR] {source} search failed: {result}")
                else:
                    all_tools.extend(result)
                    logger.info(
                        f"✅ [GEMINI RESULT] {source} returned {len(result)} results"
                    )
                    if source == "internet" and result:
                        logger.info(
                            f"🌐 [GEMINI RESULT] Gemini web search found {len(result)} tools:"
                        )
                        for idx, tool in enumerate(result, 1):
                            logger.info(
                                f"  📦 Tool #{idx}: {tool.get('Title', 'Unknown')} - {tool.get('Website', 'No URL')}"
                            )

        logger.info(f"Total results from all sources: {len(all_tools)}")
        return all_tools

    async def _filter_tools(
        self, query: str, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and select the best tools using intelligent deduplication and LLM analysis."""
        try:
            # Separate Gemini tools from other tools
            gemini_tools = [
                t for t in tools if "Internet Search" in t.get("Source", "")
            ]
            other_tools = [
                t for t in tools if "Internet Search" not in t.get("Source", "")
            ]

            logger.info(f"🌐 Found {len(gemini_tools)} Gemini internet search tools")
            logger.info(f"📦 Found {len(other_tools)} tools from other sources")

            # Step 1: Aggressive deduplication - BUT PRESERVE ALL GEMINI TOOLS
            # Deduplicate only non-Gemini tools to prevent Gemini tools from being removed
            logger.info(
                f"🔍 Starting with {len(tools)} tools ({len(gemini_tools)} Gemini, {len(other_tools)} others)"
            )
            logger.info(
                f"🌐 PRESERVING ALL {len(gemini_tools)} Gemini tools (skipping deduplication for them)"
            )

            # Deduplicate only non-Gemini tools
            deduplicated_other_tools = self._intelligent_deduplication(other_tools)
            logger.info(
                f"✅ After deduplication: {len(deduplicated_other_tools)} unique non-Gemini tools"
            )

            # Combine: ALL Gemini tools (no dedup) + deduplicated others
            deduplicated_tools = gemini_tools + deduplicated_other_tools
            logger.info(
                f"✅ Total after deduplication: {len(deduplicated_tools)} tools ({len(gemini_tools)} Gemini + {len(deduplicated_other_tools)} others)"
            )

            # Step 2: Quality filtering - SKIP FOR GEMINI TOOLS
            quality_tools = []
            for tool in deduplicated_tools:
                # ALWAYS accept Gemini internet search tools without quality checks
                if "Internet Search" in tool.get("Source", ""):
                    quality_tools.append(tool)
                    logger.info(
                        f"✅ [GEMINI] Accepting internet search tool: {tool.get('Title', 'Unknown')}"
                    )
                elif self._is_quality_tool(tool):
                    quality_tools.append(tool)
                else:
                    logger.debug(
                        f"❌ Filtered out low-quality tool: {tool.get('Title', 'Unknown')}"
                    )

            # Count Gemini tools after quality filtering
            gemini_after_quality = [
                t for t in quality_tools if "Internet Search" in t.get("Source", "")
            ]
            logger.info(
                f"✅ After quality filtering: {len(quality_tools)} high-quality tools ({len(gemini_after_quality)} Gemini)"
            )

            # Step 3: Diversity selection - SKIP FOR GEMINI TOOLS
            # Separate Gemini tools to preserve them
            gemini_quality_tools = [
                t for t in quality_tools if "Internet Search" in t.get("Source", "")
            ]
            other_quality_tools = [
                t for t in quality_tools if "Internet Search" not in t.get("Source", "")
            ]

            logger.info(
                f"🌐 Preserving {len(gemini_quality_tools)} Gemini tools from diversity filtering"
            )

            # Apply diversity only to non-Gemini tools
            diverse_other_tools = self._ensure_tool_diversity(
                other_quality_tools, query
            )

            # Combine: ALL Gemini tools + diverse other tools
            diverse_tools = gemini_quality_tools + diverse_other_tools
            logger.info(
                f"✅ After diversity selection: {len(diverse_tools)} diverse tools ({len(gemini_quality_tools)} from Gemini)"
            )

            # Step 4: LLM-based intelligent selection - PRESERVE ALL GEMINI TOOLS
            if len(diverse_tools) > 10:  # Only use LLM if we have too many tools
                # Separate again for LLM selection
                gemini_diverse = [
                    t for t in diverse_tools if "Internet Search" in t.get("Source", "")
                ]
                other_diverse = [
                    t
                    for t in diverse_tools
                    if "Internet Search" not in t.get("Source", "")
                ]

                logger.info(
                    f"🤖 Running LLM selection on {len(other_diverse)} non-Gemini tools"
                )
                logger.info(
                    f"🌐 Preserving ALL {len(gemini_diverse)} Gemini tools (no LLM filtering)"
                )

                # LLM selects from non-Gemini tools only
                selected_other_tools = await self._llm_intelligent_selection(
                    other_diverse, query
                )

                # Combine: ALL Gemini + LLM-selected others
                final_tools = gemini_diverse + selected_other_tools
                logger.info(
                    f"✅ Final tools after LLM selection: {len(gemini_diverse)} Gemini + {len(selected_other_tools)} others = {len(final_tools)} total"
                )
                logger.info(
                    f"🌐 [GEMINI PRESERVED] All {len(gemini_diverse)} Gemini tools preserved through LLM selection"
                )
            else:
                final_tools = diverse_tools
                gemini_final = [
                    t for t in final_tools if "Internet Search" in t.get("Source", "")
                ]
                logger.info(
                    f"✅ Final tools (no LLM selection needed): {len(final_tools)} total ({len(gemini_final)} Gemini)"
                )

            # Step 5: Final validation and sorting by quality
            # SEPARATE GEMINI TOOLS - NO VALIDATION OR DEDUP FOR THEM
            gemini_final_tools = [
                t for t in final_tools if "Internet Search" in t.get("Source", "")
            ]
            other_final_tools = [
                t for t in final_tools if "Internet Search" not in t.get("Source", "")
            ]

            logger.info(
                f"🌐 [GEMINI BYPASS] Skipping final dedup and quality check for {len(gemini_final_tools)} Gemini tools"
            )

            validated_tools = []
            seen_identifiers = set()

            # Process only non-Gemini tools for dedup
            for tool in other_final_tools:
                # Create multiple identifiers for final dedup check
                identifiers = self._create_tool_identifiers(tool)

                # Check if any identifier already exists
                if not any(
                    identifier in seen_identifiers for identifier in identifiers
                ):
                    # Add all identifiers to seen set
                    seen_identifiers.update(identifiers)
                    validated_tools.append(tool)
                else:
                    logger.info(
                        f"🔄 Final dedup: Skipping duplicate {tool.get('Title', 'Unknown')}"
                    )

            # Sort only non-Gemini tools by quality score
            validated_tools.sort(
                key=lambda x: self._calculate_tool_quality_score(x), reverse=True
            )

            # Combine: ALL Gemini tools (no sorting, no dedup) + validated others
            validated_tools = gemini_final_tools + validated_tools

            logger.info(
                f"✅ [FINAL] Total tools: {len(validated_tools)} ({len(gemini_final_tools)} Gemini + {len(validated_tools) - len(gemini_final_tools)} others)"
            )

            logger.info(
                f"✅ Final selection: {len(validated_tools)} unique, high-quality tools"
            )

            # Log final selection details
            for i, tool in enumerate(validated_tools, 1):
                title = tool.get("Title", "Unknown")
                website = tool.get("Website", "")
                quality = self._calculate_tool_quality_score(tool)
                logger.info(f"✅ {i}. {title} (Quality: {quality:.1f}/5.0) - {website}")

            return validated_tools

        except Exception as e:
            logger.error(f"Filter error: {e}")
            # Fallback to quality-based selection with deduplication
            return self._fallback_tool_selection_with_tools(tools, query)

    def _intelligent_deduplication(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform intelligent deduplication using multiple strategies."""
        try:
            deduplicated = []
            seen_identifiers = set()

            # Sort by quality score first (keep highest quality duplicates)
            sorted_tools = sorted(
                tools, key=lambda x: self._calculate_tool_quality_score(x), reverse=True
            )

            for tool in sorted_tools:
                identifiers = self._create_tool_identifiers(tool)

                # Check if any identifier already exists
                is_duplicate = False
                for identifier in identifiers:
                    if identifier in seen_identifiers:
                        is_duplicate = True
                        logger.debug(
                            f"🔄 Duplicate detected: {tool.get('Title', 'Unknown')} (matched: {identifier})"
                        )
                        break

                if not is_duplicate:
                    # Add all identifiers to seen set
                    seen_identifiers.update(identifiers)
                    deduplicated.append(tool)

            return deduplicated

        except Exception as e:
            logger.error(f"Error in intelligent deduplication: {e}")
            return tools

    async def _save_internet_tools_to_database(
        self, tools: List[Dict[str, Any]]
    ) -> int:
        """
        Save Internet Search tools directly to PostgreSQL Tool model.
        This allows tool chat to work immediately without waiting for scraping.

        Args:
            tools: List of tools to save (only Internet Search tools will be saved)

        Returns:
            Number of tools successfully saved
        """
        try:
            from asgiref.sync import sync_to_async
            from django.db import IntegrityError
            from django.db.models import Q

            from tools.models import Tool

            # Filter to only Internet Search tools
            internet_tools = [
                t for t in tools if "Internet Search" in t.get("Source", "")
            ]

            if not internet_tools:
                return 0

            logger.info(
                f"💾 Saving {len(internet_tools)} Internet Search tools to PostgreSQL..."
            )

            saved_count = 0
            for tool in internet_tools:
                try:
                    title = tool.get("Title", "Unknown")
                    website = tool.get("Website", "")
                    description = tool.get("Description", "")

                    # Check if tool already exists by website or title
                    existing_tool = await sync_to_async(
                        Tool.objects.filter(Q(website=website) | Q(title=title)).first
                    )()

                    if existing_tool:
                        logger.info(f"⏭️  Tool already exists: {title}")
                        continue

                    # Create new tool with redirect URL as-is
                    new_tool = await sync_to_async(Tool.objects.create)(
                        title=title,
                        website=website,  # Save redirect URL as-is
                        description=description,
                        category=tool.get("Category", "AI Tools"),
                        features=tool.get("Features", ""),
                        tags=tool.get("Tags", ""),
                        twitter=tool.get("Twitter", ""),
                        facebook=tool.get("Facebook", ""),
                        linkedin=tool.get("Linkedin", ""),
                        instagram=tool.get("Instagram", ""),
                        price_from=tool.get("Price From", ""),
                        price_to=tool.get("Price To", ""),
                    )

                    saved_count += 1
                    logger.info(
                        f"✅ Saved tool to database: {title} (ID: {new_tool.id})"
                    )

                except IntegrityError as e:
                    logger.warning(f"⚠️  Duplicate tool skipped: {title} - {e}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error saving tool {title}: {e}")
                    continue

            logger.info(
                f"💾 Successfully saved {saved_count}/{len(internet_tools)} Internet Search tools to PostgreSQL"
            )
            return saved_count

        except Exception as e:
            logger.error(f"❌ Error in _save_internet_tools_to_database: {e}")
            return 0

    def _ensure_tool_diversity(
        self, tools: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Ensure tool diversity by category and functionality."""
        try:
            if len(tools) <= 8:
                return tools  # No need to reduce if we have few tools

            # Group tools by category and functionality
            category_groups = {}
            for tool in tools:
                category = tool.get("Category", "").lower().strip()
                if not category:
                    category = "general"

                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(tool)

            # Select best tools from each category
            diverse_tools = []
            max_per_category = max(1, 8 // len(category_groups))  # Distribute evenly

            for category, category_tools in category_groups.items():
                # Sort by quality within category
                category_tools.sort(
                    key=lambda x: self._calculate_tool_quality_score(x), reverse=True
                )

                # Take best tools from this category
                selected_from_category = category_tools[:max_per_category]
                diverse_tools.extend(selected_from_category)

                logger.debug(
                    f"📊 Category '{category}': Selected {len(selected_from_category)}/{len(category_tools)} tools"
                )

            # If we still have too many, prioritize by overall quality
            if len(diverse_tools) > 10:
                diverse_tools.sort(
                    key=lambda x: self._calculate_tool_quality_score(x), reverse=True
                )
                diverse_tools = diverse_tools[:10]

            return diverse_tools

        except Exception as e:
            logger.error(f"Error ensuring tool diversity: {e}")
            return tools[:8]  # Fallback to first 8 tools

    async def _llm_intelligent_selection(
        self, tools: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Use LLM for final intelligent tool selection."""
        try:
            # Prepare concise tools data for LLM
            tools_summary = []
            for i, tool in enumerate(tools, 1):
                title = tool.get("Title", "Unknown")
                description = (
                    tool.get("Description", "")[:150] + "..."
                    if len(tool.get("Description", "")) > 150
                    else tool.get("Description", "")
                )
                category = tool.get("Category", "")
                quality = self._calculate_tool_quality_score(tool)

                tools_summary.append(
                    f"{i}. {title} (Quality: {quality:.1f}/5) - Category: {category}\n   {description}"
                )

            tools_text = "\n\n".join(tools_summary)

            prompt = f"""You are an expert workflow architect. Select the BEST 6-8 tools for this specific need.

USER NEED: "{query}"

AVAILABLE TOOLS (all are high-quality and deduplicated):
{tools_text}

SELECTION CRITERIA:
1. **PRIORITIZE "Internet Search" SOURCE** - These are freshly discovered tools from the web
2. DIRECT RELEVANCE to the user's specific need
3. COMPLEMENTARY FUNCTIONALITY (tools should work together)
4. NO FUNCTIONAL OVERLAP (avoid similar tools)
5. WORKFLOW COVERAGE (cover different aspects of the user's need)
6. QUALITY PRIORITY (higher quality scores preferred)

INSTRUCTIONS:
- Select EXACTLY 6-8 tools (no more, no less)
- **IMPORTANT**: Include AT LEAST 3-4 tools with "Internet Search" source if available
- Ensure each tool serves a DIFFERENT purpose in the workflow
- Prioritize tools that directly address "{query}"
- Avoid tools with similar functionality
- Focus on creating a complete, logical workflow

Return ONLY the tool numbers (1-based) separated by commas.
Example: 1,3,5,7,9,12
No explanations, just numbers."""

            response_text = await get_shared_llm().generate_response(prompt)
            cleaned_response = response_text.replace('"', "").replace("'", "").strip()

            # Parse selected indices
            selected_indices = []
            for num_str in cleaned_response.split(","):
                try:
                    idx = int(num_str.strip()) - 1
                    if 0 <= idx < len(tools):
                        selected_indices.append(idx)
                except ValueError:
                    continue

            if selected_indices:
                selected_tools = [tools[i] for i in selected_indices]
                logger.info(
                    f"🤖 LLM selected {len(selected_tools)} tools from {len(tools)} options"
                )
                return selected_tools
            else:
                logger.warning("LLM selection failed, using top quality tools")
                return tools[:8]

        except Exception as e:
            logger.error(f"Error in LLM intelligent selection: {e}")
            return tools[:8]

    def _prepare_enhanced_tools_data_for_llm(self, tools: List[Dict[str, Any]]) -> str:
        """Prepare enhanced tools data with detailed descriptions for LLM analysis."""
        try:
            tools_text = []

            for i, tool in enumerate(tools, 1):
                title = tool.get("Title", "Unknown Tool")
                description = tool.get("Description", "") or tool.get(
                    "_detailed_description", ""
                )
                features = tool.get("Features", "") or tool.get(
                    "_detailed_features", ""
                )
                category = tool.get("Category", "")
                website = tool.get("Website", "")
                source = tool.get("Source", "Unknown")

                # Create enhanced tool description
                tool_info = f"{i}. **{title}**"

                if description:
                    tool_info += f"\n   Description: {description[:300]}..."

                if features:
                    tool_info += f"\n   Key Features: {features[:200]}..."

                if category:
                    tool_info += f"\n   Category: {category}"

                if website:
                    tool_info += f"\n   Website: {website}"
                else:
                    tool_info += f"\n   Website: ❌ No website available"

                tool_info += f"\n   Source: {source}"

                # Add quality indicators
                quality_score = self._calculate_tool_quality_score(tool)
                tool_info += f"\n   Quality Score: {quality_score:.1f}/5.0"

                tools_text.append(tool_info)

            return "\n\n".join(tools_text)

        except Exception as e:
            logger.error(f"Error preparing enhanced tools data: {e}")
            return ToolDataFormatter.prepare_tools_data_for_prompt(tools)

    def _calculate_tool_quality_score(self, tool: Dict[str, Any]) -> float:
        """Calculate quality score for a tool based on available data."""
        try:
            score = 0.0

            # Website presence (2.0 points)
            website = tool.get("Website", "")
            if (
                website
                and website.startswith(("http://", "https://"))
                and "example.com" not in website
            ):
                score += 2.0

            # Description quality (1.5 points)
            description = tool.get("Description", "") or tool.get(
                "_detailed_description", ""
            )
            if description and len(description) > 50:
                score += 1.5
            elif description and len(description) > 20:
                score += 0.8

            # Features available (1.0 points)
            features = tool.get("Features", "") or tool.get("_detailed_features", "")
            if features and len(features) > 30:
                score += 1.0
            elif features:
                score += 0.5

            # PostgreSQL enrichment (0.5 points)
            if tool.get("_postgres_enriched"):
                score += 0.5

            return min(score, 5.0)  # Cap at 5.0

        except Exception:
            return 1.0  # Default score

    def _is_quality_tool(self, tool: Dict[str, Any]) -> bool:
        """Check if a tool meets quality criteria."""
        try:
            # Must have a valid website
            website = tool.get("Website", "")
            if not website or not website.startswith(("http://", "https://")):
                return False

            # Must not be example.com or placeholder
            if "example.com" in website.lower():
                return False

            # Must have a meaningful title
            title = tool.get("Title", "")
            if not title or title.lower().strip() in [
                "tool",
                "unknown tool",
                "untitled",
                "",
            ]:
                return False

            # Quality score threshold
            quality_score = self._calculate_tool_quality_score(tool)
            return quality_score >= 2.0

        except Exception:
            return False

    def _fallback_tool_selection(
        self, tools: List[Dict[str, Any]], query: str
    ) -> List[int]:
        """Fallback tool selection based on quality scores and relevance."""
        try:
            # Score tools by quality and relevance
            scored_tools = []
            query_lower = query.lower()

            for i, tool in enumerate(tools):
                if not self._is_quality_tool(tool):
                    continue

                quality_score = self._calculate_tool_quality_score(tool)

                # Calculate relevance score
                relevance_score = 0.0
                title = tool.get("Title", "").lower()
                description = tool.get("Description", "").lower()

                # Simple keyword matching for relevance
                for word in query_lower.split():
                    if len(word) > 2:  # Skip short words
                        if word in title:
                            relevance_score += 2.0
                        if word in description:
                            relevance_score += 1.0

                total_score = quality_score + relevance_score
                scored_tools.append((i, total_score))

            # Sort by total score and return top indices
            scored_tools.sort(key=lambda x: x[1], reverse=True)
            return [idx for idx, score in scored_tools[:8]]

        except Exception as e:
            logger.error(f"Error in fallback selection: {e}")
            return list(range(min(5, len(tools))))

    def _fallback_tool_selection_with_tools(
        self, tools: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Fallback selection returning actual tools."""
        try:
            indices = self._fallback_tool_selection(tools, query)
            return [tools[i] for i in indices if 0 <= i < len(tools)]
        except Exception:
            return tools[:5]

    async def _generate_workflow(
        self, query: str, tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate workflow using existing prompt service."""
        try:
            if not tools:
                return None

            logger.info(f"Generating workflow for {len(tools)} tools")

            # Prepare tools data for the prompt
            tools_data = ToolDataFormatter.prepare_tools_data_for_prompt(tools)

            # Create workflow prompt
            prompt_template = self._create_workflow_prompt()
            prompt = prompt_template.format(task=query, tools=tools_data)

            # Get workflow from LLM
            response_text = await get_shared_llm().generate_response(prompt)

            # Parse JSON response
            try:
                workflow_data = await get_shared_llm().parse_json_response(
                    response_text
                )
                logger.info("Successfully parsed LLM workflow response")

                # CRITICAL FIX: Convert hardcoded IDs to UUIDs
                workflow_data = self._convert_hardcoded_ids_to_uuids(workflow_data)
                logger.info("Converted hardcoded IDs to UUIDs")

                return workflow_data

            except Exception as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                return self._create_fallback_workflow(tools)

        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return self._create_fallback_workflow(tools)

    def _create_fallback_workflow(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a complex fallback workflow with many nodes and edges."""
        # Separate Internet Search tools from others
        internet_search_tools = [
            tool for tool in tools if "Internet Search" in tool.get("Source", "")
        ]
        other_tools = [
            tool for tool in tools if "Internet Search" not in tool.get("Source", "")
        ]

        # For Internet Search tools: Accept ALL without website validation
        # For other tools: Filter to only include those with valid websites
        valid_other_tools = [
            tool for tool in other_tools if self._tool_has_valid_website(tool)
        ]

        # Combine: ALL Internet Search tools + valid other tools
        tools_with_websites = internet_search_tools + valid_other_tools

        if not tools_with_websites:
            logger.warning("No tools with websites found for fallback workflow")
            return {
                "query": "Generated workflow",
                "nodes": [],
                "edges": [],
            }

        # Use ALL tools (not just 12) - especially important for Internet Search tools
        selected_tools = tools_with_websites  # Use ALL tools
        logger.info(
            f"🔨 Creating fallback workflow with {len(selected_tools)} tools ({len(internet_search_tools)} from Internet Search)"
        )

        nodes = self._create_complex_workflow_nodes(selected_tools)
        edges = self._create_complex_workflow_edges(nodes)

        logger.info(
            f"✅ Created complex fallback workflow with {len(nodes)} nodes and {len(edges)} edges"
        )
        return {
            "query": "Generated complex workflow",
            "nodes": nodes,
            "edges": edges,
        }

    def _tool_has_valid_website(self, tool: Dict[str, Any]) -> bool:
        """Check if tool has a valid website."""
        website = tool.get("Website", "")
        return (
            website
            and isinstance(website, str)
            and website.startswith(("http://", "https://"))
            and "example.com" not in website.lower()
        )

    def _create_complex_workflow_nodes(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create complex workflow nodes with multiple phases and roles, ensuring no duplicates."""
        nodes = []
        workflow_phases = [
            "input",
            "processing",
            "action",
            "monitoring",
            "optimization",
        ]
        seen_tool_identifiers = set()

        # Filter and deduplicate tools before creating nodes
        unique_tools = []
        for tool in tools:
            tool_name = tool.get("Title", "")
            website = tool.get("Website", "")

            # Create identifier for deduplication
            identifier = self._create_node_identifier(tool_name, website)

            # Skip if already seen or invalid
            if (
                identifier in seen_tool_identifiers
                or not tool_name
                or not self._tool_has_valid_website(tool)
            ):
                logger.warning(f"❌ Skipping duplicate/invalid tool: '{tool_name}'")
                continue

            seen_tool_identifiers.add(identifier)
            unique_tools.append(tool)

        logger.info(f"✅ Creating workflow nodes for {len(unique_tools)} unique tools")

        # Distribute unique tools across phases
        tools_per_phase = max(1, len(unique_tools) // len(workflow_phases))

        for i, tool in enumerate(unique_tools):
            node_id = str(uuid.uuid4())
            tool_name = tool.get("Title", "")
            tool_desc = tool.get("Description", "") or tool.get(
                "_detailed_description", ""
            )

            # Determine workflow phase
            phase_index = min(i // tools_per_phase, len(workflow_phases) - 1)
            workflow_phase = workflow_phases[phase_index]

            # Determine criticality based on phase and position
            if workflow_phase in ["processing", "action"]:
                criticality = "high"
            elif workflow_phase in ["input", "monitoring"]:
                criticality = "medium"
            else:
                criticality = "low"

            # Generate phase-specific recommendation reason
            recommendation_reason = self._generate_phase_specific_recommendation(
                tool_name, tool_desc, workflow_phase, i + 1, len(unique_tools)
            )

            # Calculate position in complex grid layout
            phase_offset = phase_index * 400  # Horizontal spacing between phases
            vertical_offset = (
                i % tools_per_phase
            ) * 200  # Vertical spacing within phase

            node = {
                "id": node_id,
                "type": "tool",
                "data": {
                    "label": tool_name,
                    "description": tool_desc,
                    "features": (
                        tool.get("Features", "").split(",")
                        if tool.get("Features")
                        else tool.get("_detailed_features", "").split(",")
                        if tool.get("_detailed_features")
                        else []
                    ),
                    "tags": (
                        tool.get("Category", "").split(",")
                        if tool.get("Category")
                        else []
                    ),
                    "recommendation_reason": recommendation_reason,
                    "workflow_phase": workflow_phase,
                    "criticality": criticality,
                    "connections_count": 0,  # Will be updated when creating edges
                    "website": tool.get("Website", ""),
                    "twitter": tool.get("Twitter", ""),
                    "facebook": tool.get("Facebook", ""),
                    "linkedin": tool.get("LinkedIn", ""),
                    "instagram": tool.get("Instagram", ""),
                    # Add unique identifier for tracking
                    "unique_identifier": self._create_node_identifier(
                        tool_name, tool.get("Website", "")
                    ),
                },
                "position": {"x": 100 + phase_offset, "y": 100 + vertical_offset},
            }
            nodes.append(node)

            logger.info(
                f"✅ Created {workflow_phase} tool node: {tool_name} ({criticality} criticality)"
            )

        return nodes

    def _create_node_identifier(self, tool_name: str, website: str) -> str:
        """Create a unique identifier for workflow nodes to prevent duplicates."""
        try:
            # Check if this is an Internet Search redirect URL
            is_redirect_url = website and (
                "redirect" in website
                or "grounding-api" in website
                or "vertexaisearch" in website
            )

            # For Internet Search tools with redirect URLs, ALWAYS use tool name (not domain)
            if is_redirect_url:
                if tool_name:
                    normalized_name = (
                        tool_name.lower()
                        .replace(" ", "")
                        .replace("-", "")
                        .replace("_", "")
                        .replace(".", "")
                        .strip()
                    )
                    if len(normalized_name) > 2:
                        return f"name:{normalized_name}"
                return f"original:{tool_name.lower()}" if tool_name else "unknown"

            # For non-redirect URLs: Use website domain if available
            if website and website.startswith(("http://", "https://")):
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(website).netloc.lower()
                    if domain:
                        return f"domain:{domain}"
                except Exception:
                    pass

            # Fallback: Use normalized tool name
            if tool_name:
                normalized_name = (
                    tool_name.lower()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                    .replace(".", "")
                    .replace("ai", "")
                    .strip()
                )
                if len(normalized_name) > 3:
                    return f"name:{normalized_name}"

            # Last resort: Use original name
            return f"original:{tool_name.lower()}" if tool_name else "unknown"

        except Exception as e:
            logger.error(f"Error creating node identifier: {e}")
            return f"fallback:{tool_name.lower()}" if tool_name else "error"

    def _generate_phase_specific_recommendation(
        self, tool_name: str, tool_desc: str, phase: str, position: int, total: int
    ) -> str:
        """Generate phase-specific recommendation reasons."""
        phase_descriptions = {
            "input": f"Essential for data collection and input processing in the initial phase of your workflow. {tool_name} serves as a critical entry point for gathering and organizing information.",
            "processing": f"Core processing engine that analyzes and transforms data from input tools. {tool_name} performs critical computations and decision-making logic in your workflow.",
            "action": f"Execution powerhouse that implements decisions and automates actions based on processed data. {tool_name} delivers tangible results and performs key workflow operations.",
            "monitoring": f"Vigilant oversight system that tracks performance and ensures workflow quality. {tool_name} provides essential monitoring and feedback capabilities.",
            "optimization": f"Continuous improvement engine that enhances workflow efficiency over time. {tool_name} analyzes patterns and optimizes future workflow executions.",
        }

        base_reason = phase_descriptions.get(
            phase, f"Important component in your workflow automation strategy."
        )

        # Add tool-specific context if description is available
        if tool_desc and len(tool_desc) > 20:
            context_keywords = [
                "automate",
                "manage",
                "track",
                "analyze",
                "optimize",
                "integrate",
            ]
            found_keywords = [kw for kw in context_keywords if kw in tool_desc.lower()]
            if found_keywords:
                base_reason += f" Specifically chosen for its {', '.join(found_keywords)} capabilities."

        return base_reason

    def _generate_simple_recommendation(
        self, tool_name: str, tool_description: str, position: int, total: int
    ) -> str:
        """Generate a simple recommendation reason for fallback workflows."""
        # Extract key phrases from description for more context
        desc_lower = tool_description.lower() if tool_description else ""

        # Detect key capabilities from description
        capability = "core functionality"
        if "automate" in desc_lower or "automation" in desc_lower:
            capability = "automation capabilities"
        elif "manage" in desc_lower or "management" in desc_lower:
            capability = "management features"
        elif "analyze" in desc_lower or "analytics" in desc_lower:
            capability = "analytics capabilities"

        # Position-based reasoning
        if position == 1:
            return f"Selected as the primary tool to begin your workflow based on its {capability}"
        elif position == total:
            return f"Chosen to complete your workflow with essential final-stage {capability}"
        else:
            return f"Included for its complementary {capability} that enhance your workflow's effectiveness"

    def _create_complex_workflow_edges(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create complex workflow edges with multiple connection patterns."""
        edges = []

        if len(nodes) < 2:
            logger.info("Not enough nodes for complex edges")
            return edges

        # Group nodes by phase for intelligent connections
        phase_groups = {}
        for node in nodes:
            phase = node.get("data", {}).get("workflow_phase", "processing")
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(node)

        # Pattern 1: Sequential phase connections (input → processing → action → monitoring → optimization)
        phase_order = ["input", "processing", "action", "monitoring", "optimization"]
        edges.extend(self._create_phase_to_phase_edges(phase_groups, phase_order))

        # Pattern 2: Parallel connections within phases
        for phase, phase_nodes in phase_groups.items():
            if len(phase_nodes) > 1:
                edges.extend(self._create_parallel_edges(phase_nodes, phase))

        # Pattern 3: Hub connections (high criticality tools connect to many others)
        hub_nodes = [n for n in nodes if n.get("data", {}).get("criticality") == "high"]
        edges.extend(self._create_hub_connections(hub_nodes, nodes))

        # Pattern 4: Feedback loops (monitoring/optimization back to input/processing)
        edges.extend(self._create_feedback_loops(phase_groups))

        # Pattern 5: Conditional branches (processing tools to multiple action tools)
        edges.extend(self._create_conditional_branches(phase_groups))

        # Update connection counts in node data
        self._update_node_connection_counts(nodes, edges)

        logger.info(
            f"✅ Created complex workflow with {len(edges)} edges across {len(nodes)} nodes"
        )
        return edges

    def _create_phase_to_phase_edges(
        self, phase_groups: Dict, phase_order: List[str]
    ) -> List[Dict[str, Any]]:
        """Create edges between different workflow phases."""
        edges = []

        for i in range(len(phase_order) - 1):
            current_phase = phase_order[i]
            next_phase = phase_order[i + 1]

            current_nodes = phase_groups.get(current_phase, [])
            next_nodes = phase_groups.get(next_phase, [])

            if current_nodes and next_nodes:
                # Connect each node in current phase to nodes in next phase
                for current_node in current_nodes:
                    for next_node in next_nodes[:2]:  # Limit connections per node
                        edge = self._create_edge(
                            current_node,
                            next_node,
                            "primary",
                            f"Phase transition: {current_phase} to {next_phase}",
                        )
                        edges.append(edge)

        return edges

    def _create_parallel_edges(
        self, phase_nodes: List[Dict], phase: str
    ) -> List[Dict[str, Any]]:
        """Create parallel connections within a phase."""
        edges = []

        if len(phase_nodes) > 2:
            # Create mesh connections within phase
            for i, node1 in enumerate(phase_nodes):
                for j, node2 in enumerate(phase_nodes[i + 1 :], i + 1):
                    if j - i <= 2:  # Only connect to nearby nodes
                        edge = self._create_edge(
                            node1,
                            node2,
                            "secondary",
                            f"Parallel processing within {phase} phase",
                        )
                        edges.append(edge)

        return edges

    def _create_hub_connections(
        self, hub_nodes: List[Dict], all_nodes: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Create hub connections for high criticality tools."""
        edges = []

        for hub_node in hub_nodes:
            hub_phase = hub_node.get("data", {}).get("workflow_phase", "")
            connected_count = 0

            for other_node in all_nodes:
                if (
                    other_node["id"] != hub_node["id"] and connected_count < 4
                ):  # Max 4 connections per hub
                    other_phase = other_node.get("data", {}).get("workflow_phase", "")

                    # Connect hub to nodes in adjacent phases
                    if other_phase != hub_phase:
                        edge = self._create_edge(
                            hub_node,
                            other_node,
                            "primary",
                            f"Hub connection from {hub_phase} to {other_phase}",
                        )
                        edges.append(edge)
                        connected_count += 1

        return edges

    def _create_feedback_loops(self, phase_groups: Dict) -> List[Dict[str, Any]]:
        """Create feedback loops from later phases to earlier phases."""
        edges = []

        monitoring_nodes = phase_groups.get("monitoring", [])
        optimization_nodes = phase_groups.get("optimization", [])
        input_nodes = phase_groups.get("input", [])
        processing_nodes = phase_groups.get("processing", [])

        # Monitoring feedback to input
        for monitor_node in monitoring_nodes[:2]:
            for input_node in input_nodes[:1]:
                edge = self._create_edge(
                    monitor_node,
                    input_node,
                    "feedback",
                    "Monitoring feedback to improve input quality",
                )
                edges.append(edge)

        # Optimization feedback to processing
        for opt_node in optimization_nodes[:2]:
            for proc_node in processing_nodes[:1]:
                edge = self._create_edge(
                    opt_node,
                    proc_node,
                    "feedback",
                    "Optimization feedback to enhance processing",
                )
                edges.append(edge)

        return edges

    def _create_conditional_branches(self, phase_groups: Dict) -> List[Dict[str, Any]]:
        """Create conditional branches for decision points."""
        edges = []

        processing_nodes = phase_groups.get("processing", [])
        action_nodes = phase_groups.get("action", [])

        # Each processing node can trigger multiple action nodes
        for proc_node in processing_nodes:
            for action_node in action_nodes[:3]:  # Up to 3 conditional branches
                edge = self._create_edge(
                    proc_node,
                    action_node,
                    "conditional",
                    "Conditional execution based on processing results",
                )
                edges.append(edge)

        return edges

    def _create_edge(
        self, source_node: Dict, target_node: Dict, flow_type: str, description: str
    ) -> Dict[str, Any]:
        """Create a single edge with proper metadata."""
        source_name = source_node.get("data", {}).get("label", "Unknown")
        target_name = target_node.get("data", {}).get("label", "Unknown")

        return {
            "id": str(uuid.uuid4()),
            "source": source_node["id"],
            "target": target_node["id"],
            "type": "default",
            "data": {
                "label": f"{source_name} → {target_name}",
                "description": description,
                "flow_type": flow_type,
            },
        }

    def _update_node_connection_counts(
        self, nodes: List[Dict], edges: List[Dict]
    ) -> None:
        """Update connection counts in node data."""
        connection_counts = {}

        for edge in edges:
            source_id = edge["source"]
            target_id = edge["target"]

            connection_counts[source_id] = connection_counts.get(source_id, 0) + 1
            connection_counts[target_id] = connection_counts.get(target_id, 0) + 1

        for node in nodes:
            node_id = node["id"]
            node["data"]["connections_count"] = connection_counts.get(node_id, 0)

    def _convert_hardcoded_ids_to_uuids(
        self, workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert hardcoded node/edge IDs (like node_001) to proper UUIDs."""
        try:
            # Create mapping from old IDs to new UUIDs
            id_mapping = {}

            # Process nodes first
            nodes = workflow_data.get("nodes", [])
            for node in nodes:
                old_id = node.get("id", "")
                if old_id and (
                    old_id.startswith("node_") or old_id.startswith("tool_")
                ):
                    # Generate new UUID for hardcoded ID
                    new_uuid = str(uuid.uuid4())
                    id_mapping[old_id] = new_uuid
                    node["id"] = new_uuid
                    logger.info(f"🔄 Converted node ID: {old_id} → {new_uuid[:8]}...")
                elif not old_id:
                    # Generate UUID for missing ID
                    new_uuid = str(uuid.uuid4())
                    node["id"] = new_uuid
                    logger.info(
                        f"✅ Generated UUID for node missing ID: {new_uuid[:8]}..."
                    )

            # Process edges and update their source/target references
            edges = workflow_data.get("edges", [])
            for edge in edges:
                old_edge_id = edge.get("id", "")
                if (old_edge_id and old_edge_id.startswith("edge_")) or (
                    not old_edge_id
                ):
                    # Generate new UUID for hardcoded or missing edge ID
                    edge["id"] = str(uuid.uuid4())

                # Update source and target references
                old_source = edge.get("source", "")
                old_target = edge.get("target", "")

                if old_source in id_mapping:
                    edge["source"] = id_mapping[old_source]
                    logger.info(
                        f"🔗 Updated edge source: {old_source} → {id_mapping[old_source][:8]}..."
                    )

                if old_target in id_mapping:
                    edge["target"] = id_mapping[old_target]
                    logger.info(
                        f"🔗 Updated edge target: {old_target} → {id_mapping[old_target][:8]}..."
                    )

            logger.info(f"✅ Converted {len(id_mapping)} hardcoded IDs to UUIDs")
            return workflow_data

        except Exception as e:
            logger.error(f"❌ Error converting hardcoded IDs: {e}")
            # Return original data if conversion fails
            return workflow_data

    def _create_workflow_prompt(self) -> ChatPromptTemplate:
        """Create the workflow generation prompt template for complex workflows."""
        return ChatPromptTemplate.from_template(
            """
You are a WORKFLOW AUTOMATION ARCHITECT who creates COMPLEX, SOPHISTICATED workflows with MANY interconnected tools.

### Task:
Create a COMPREHENSIVE, COMPLEX workflow for: {task}

**Available Tools:** {tools}

### ⚠️ CRITICAL REQUIREMENT - INCLUDE ALL TOOLS:
**YOU MUST INCLUDE ALL OR MOST OF THE AVAILABLE TOOLS IN THE WORKFLOW!**
- If 20 tools are provided, create a workflow with AT LEAST 15-20 nodes
- If 25 tools are provided, create a workflow with AT LEAST 20-25 nodes
- DO NOT create workflows with only 1-3 nodes when many tools are available
- EVERY tool provided should be considered for inclusion

### COMPLEXITY REQUIREMENTS:
1. **MAXIMUM TOOLS**: Use ALL or MOST of the available tools (aim for 8-15 nodes minimum)
2. **COMPLEX CONNECTIONS**: Create MANY edges - each tool should connect to 2-4 other tools
3. **PARALLEL PROCESSING**: Multiple tools should work simultaneously
4. **BRANCHING LOGIC**: Create conditional paths and decision points
5. **FEEDBACK LOOPS**: Some tools should feed back to earlier stages
6. **MULTI-STAGE WORKFLOW**: Create distinct phases with multiple tools per phase
7. **REDUNDANCY & BACKUP**: Include alternative paths for critical functions

### ADVANCED CONNECTION PATTERNS:
- **Hub Pattern**: Central tools that connect to many others (e.g., CRM connecting to 5+ tools)
- **Pipeline Pattern**: Sequential processing chains with parallel branches
- **Mesh Pattern**: Tools interconnected in complex webs of data flow
- **Feedback Pattern**: Output from later stages feeding back to earlier ones
- **Conditional Pattern**: Different tools activated based on conditions

### WORKFLOW PHASES TO IMPLEMENT:
1. **INPUT PHASE**: Multiple data collection and input tools
2. **PROCESSING PHASE**: Analysis, transformation, and decision tools
3. **ACTION PHASE**: Execution and automation tools
4. **MONITORING PHASE**: Tracking and feedback tools
5. **OPTIMIZATION PHASE**: Continuous improvement tools

### EDGE CREATION RULES:
- Each tool should have 2-4 connections (both incoming and outgoing)
- Create logical data flows between related tools
- Add feedback loops for continuous improvement
- Include parallel processing paths
- Create conditional branches for different scenarios

### Response Format:
Return ONLY valid JSON with this EXACT structure:

{{
    "query": "Original user query",
    "nodes": [
        {{
            "id": "node_001",
            "type": "tool",
            "data": {{
                "label": "Tool Name",
                "description": "Tool description",
                "features": ["feature1", "feature2"],
                "tags": ["tag1", "tag2"],
                "recommendation_reason": "Explain this tool's SPECIFIC ROLE in this complex workflow, its connections to other tools, and its critical contribution to the overall automation strategy.",
                "website": "",
                "twitter": "",
                "facebook": "",
                "linkedin": "",
                "instagram": "",
                "workflow_phase": "input|processing|action|monitoring|optimization",
                "connections_count": 3,
                "criticality": "high|medium|low"
            }}
        }}
    ],
    "edges": [
        {{
            "id": "edge_001",
            "source": "node_001",
            "target": "node_002",
            "type": "default",
            "data": {{
                "label": "Data Flow Description",
                "description": "What data/information flows between these tools",
                "flow_type": "primary|secondary|feedback|conditional"
            }}
        }}
    ]
}}

**CRITICAL COMPLEXITY REQUIREMENTS**:
- **MINIMUM 8+ NODES**: Use most/all available tools (but ensure quality)
- **MINIMUM 12+ EDGES**: Each tool connects to multiple others
- **MULTIPLE PARALLEL PATHS**: Tools working simultaneously
- **FEEDBACK LOOPS**: At least 2-3 feedback connections
- **CONDITIONAL BRANCHES**: Different paths for different scenarios
- **PHASE DISTRIBUTION**: Tools distributed across all 5 workflow phases
- **HUB TOOLS**: 2-3 central tools with 4+ connections each

**STRICT DEDUPLICATION REQUIREMENTS**:
- **NO DUPLICATE TOOLS**: Each tool must be UNIQUE - never include the same tool twice
- **NO SIMILAR TOOLS**: Avoid tools with identical or very similar functionality
- **DIVERSE FUNCTIONALITY**: Each tool should serve a DIFFERENT purpose in the workflow
- **UNIQUE WEBSITES**: Each tool must have a different website/domain

**🚨 CRITICAL: INTERNET SEARCH TOOLS ARE MANDATORY 🚨**:
- **INCLUDE EVERY SINGLE TOOL FROM "Internet Search"**: Any tool with source containing "Internet Search" MUST be included in the workflow - NO EXCEPTIONS
- **NO FILTERING FOR INTERNET SEARCH TOOLS**: Internet Search tools should NEVER be filtered out, regardless of website format
- **INTERNET SEARCH TOOLS ARE MANDATORY**: These are freshly discovered tools and EVERY ONE must be in the workflow
- **ACCEPT ALL WEBSITE FORMATS FOR INTERNET SEARCH**: Internet Search tools may have redirect URLs or non-standard formats - include them anyway
- **COUNT CHECK**: If you receive 20 Internet Search tools, your workflow MUST have AT LEAST 20 nodes (one for each Internet Search tool)

**WEBSITE VALIDATION FOR NON-INTERNET-SEARCH TOOLS ONLY**:
- For tools NOT from Internet Search: Each tool should have a valid website starting with http:// or https://
- For tools FROM Internet Search: Accept ANY website format (including redirect URLs)

Create the MOST COMPLEX workflow possible while maintaining logical connections and real-world feasibility. PRIORITIZE including ALL Internet Search tools.

"""
        )

    async def generate_workflow(
        self, task: str, tools: list, refined_query: str = None
    ) -> dict:
        """Generate a sequential workflow with proper execution flow.

        Args:
            task: The task description
            tools: List of available tools
            refined_query: Optional refined query for better recommendation reasons

        Returns:
            Generated workflow as dictionary
        """
        try:
            logger.info(f"🔨 Generating sequential workflow for {len(tools)} tools")

            # Use the new sequential workflow generator
            from ai_tool_recommender.services.sequential_workflow_generator import (
                SequentialWorkflowGenerator,
            )

            workflow_generator = SequentialWorkflowGenerator()
            workflow = await workflow_generator.generate_sequential_workflow(
                tools=tools,
                refined_query=refined_query or task,
                original_query=task,
            )

            if workflow and workflow.get("nodes"):
                logger.info(
                    f"✅ Generated sequential workflow with {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges"
                )
                return workflow
            else:
                logger.warning("Sequential workflow generation failed, using fallback")
                return self._create_fallback_workflow(tools)

        except Exception as e:
            logger.error(f"❌ Error generating workflow: {e}", exc_info=True)
            return self._create_fallback_workflow(tools)

    async def _ensure_recommendation_reasons(
        self, workflow: dict, task: str, refined_query: str = None
    ) -> dict:
        """
        Generate contextual recommendation reasons using LLM.
        Explains why THIS tool was selected for THIS specific workflow goal.

        Args:
            workflow: Workflow dictionary with nodes
            task: User's original task/query
            refined_query: Refined query (preferred if available)

        Returns:
            Updated workflow with recommendation_reason on all nodes
        """
        try:
            nodes = workflow.get("nodes", [])
            if not nodes:
                return workflow

            # Use refined_query if available, otherwise fall back to task
            query_to_analyze = refined_query or task
            if not query_to_analyze:
                return workflow

            # Generate reasons for each node
            for i, node in enumerate(nodes):
                node_data = node.get("data", {})
                tool_name = node_data.get("label", "Tool")

                # Skip trigger nodes
                if node.get("type") == "trigger" or tool_name.lower() in [
                    "trigger",
                    "start",
                    "begin",
                ]:
                    continue

                # Skip if reason already exists and is sufficient
                existing_reason = node_data.get("recommendation_reason", "")
                if existing_reason and len(existing_reason.strip()) >= 30:
                    continue

                # Extract tool information
                tool_desc = node_data.get("description", "")
                tool_features = node_data.get("features", [])
                tool_tags = node_data.get("tags", [])

                # Extract workflow context
                previous_tools = [
                    nodes[j].get("data", {}).get("label", "")
                    for j in range(i)
                    if nodes[j].get("type") != "trigger"
                ][
                    -2:
                ]  # Last 2 previous tools

                next_tools = [
                    nodes[j].get("data", {}).get("label", "")
                    for j in range(i + 1, len(nodes))
                    if nodes[j].get("type") != "trigger"
                ][
                    :2
                ]  # Next 2 tools

                # Generate contextual reason using LLM
                reason = await self._generate_contextual_reason(
                    tool_name=tool_name,
                    tool_desc=tool_desc,
                    tool_features=tool_features,
                    tool_tags=tool_tags,
                    refined_query=query_to_analyze,
                    position=i + 1,
                    total_nodes=len(nodes),
                    previous_tools=previous_tools,
                    next_tools=next_tools,
                )

                if reason:
                    node_data["recommendation_reason"] = reason
                    logger.info(f"✅ Generated contextual reason for {tool_name}")

            return workflow

        except Exception as e:
            logger.error(f"Error ensuring recommendation reasons: {e}")
            return workflow

    async def _generate_contextual_reason(
        self,
        tool_name: str,
        tool_desc: str,
        tool_features: list,
        tool_tags: list,
        refined_query: str,
        position: int,
        total_nodes: int,
        previous_tools: list,
        next_tools: list,
    ) -> str:
        """Generate contextual recommendation reason using LLM."""
        try:
            # Build context string
            features_str = ", ".join(tool_features[:5]) if tool_features else "N/A"
            tags_str = ", ".join(tool_tags[:5]) if tool_tags else "N/A"
            prev_tools_str = ", ".join(previous_tools) if previous_tools else "None"
            next_tools_str = ", ".join(next_tools) if next_tools else "None"

            prompt = f"""User's Goal: "{refined_query}"

Tool Information:
- Name: {tool_name}
- Description: {tool_desc[:300] if tool_desc else "N/A"}
- Features: {features_str}
- Tags: {tags_str}

Workflow Context:
- Position: {position} of {total_nodes}
- Previous tools: {prev_tools_str}
- Next tools: {next_tools_str}

Task: Generate a recommendation reason (2-3 sentences) explaining why THIS tool is perfect for the user's goal.

CRITICAL RULES:
1. NEVER say "I cannot provide a recommendation", "description is incomplete", or "insufficient information".
2. If the tool description is N/A or sparse, use your INTERNAL KNOWLEDGE about "{tool_name}" to explain its value.
3. Be EXTREMELY POSITIVE and SPECIFIC about how it helps with "{refined_query}".
4. Focus on the actual connection between the tool and the goal.

Your response:"""

            reason = await get_shared_llm().generate_response(prompt)
            reason = reason.strip() if reason else None

            # Defensive check for refusals
            refusal_keywords = [
                "cannot provide",
                "insufficient",
                "incomplete",
                "description is too short",
                "not enough information",
            ]
            if reason and any(kw in reason.lower() for kw in refusal_keywords):
                logger.warning(
                    f"⚠️ LLM tried to refuse recommendation for {tool_name}. Applying fallback."
                )
                return f"Selected because {tool_name} provides essential capabilities to achieve your goal of '{refined_query}', serving as a key component in this automated workflow."

            return reason

        except Exception as e:
            logger.error(f"Error generating contextual reason: {e}")
            return None

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response and extract valid JSON workflow."""
        try:
            # Clean the response first
            cleaned_response = response.strip()

            # Remove markdown code blocks if present
            if "```json" in cleaned_response:
                cleaned_response = (
                    cleaned_response.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_response:
                cleaned_response = (
                    cleaned_response.split("```")[1].split("```")[0].strip()
                )

            # Find JSON content between first { and last }
            start_idx = cleaned_response.find("{")
            end_idx = cleaned_response.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = cleaned_response[start_idx : end_idx + 1]
            else:
                json_content = cleaned_response

            # Try to parse the JSON
            workflow = json.loads(json_content)

            # Validate the workflow structure
            if not isinstance(workflow, dict):
                return None

            if (
                "query" not in workflow
                or "nodes" not in workflow
                or "edges" not in workflow
            ):
                return None

            if not isinstance(workflow["nodes"], list) or not isinstance(
                workflow["edges"], list
            ):
                return None

            # Validate node structure
            for node in workflow["nodes"]:
                if (
                    not isinstance(node, dict)
                    or "id" not in node
                    or "type" not in node
                    or "data" not in node
                ):
                    return None
                if not isinstance(node["data"], dict) or "label" not in node["data"]:
                    return None

            # Validate edge structure
            for edge in workflow["edges"]:
                if (
                    not isinstance(edge, dict)
                    or "id" not in edge
                    or "source" not in edge
                    or "target" not in edge
                    or "type" not in edge
                ):
                    return None

            return workflow

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None

    def _map_llm_nodes_to_real_tools(self, workflow: dict, real_tools: list) -> dict:
        """
        Map LLM-generated nodes back to real tools by matching names/descriptions.
        This ensures the workflow uses REAL tool data (URLs, etc.) instead of fake example.com URLs.

        Args:
            workflow: LLM-generated workflow with potentially fake nodes
            real_tools: List of real tools with actual data

        Returns:
            Workflow with nodes mapped to real tools
        """
        try:
            nodes = workflow.get("nodes", [])
            mapped_nodes = []

            for llm_node in nodes:
                node_data = llm_node.get("data", {})
                llm_label = node_data.get("label", "").lower().strip()

                # Try to find matching real tool
                matched_tool = None
                best_match_score = 0

                for real_tool in real_tools:
                    real_title = real_tool.get("Title", "").lower().strip()
                    real_description = real_tool.get("Description", "").lower()

                    # Calculate match score
                    score = 0
                    if llm_label == real_title:
                        score = 100  # Exact match
                    elif llm_label in real_title or real_title in llm_label:
                        score = 80  # Partial match
                    elif (
                        llm_label in real_description
                        or real_description
                        and any(
                            word in real_description
                            for word in llm_label.split()
                            if len(word) > 3
                        )
                    ):
                        score = 50  # Description match

                    if score > best_match_score:
                        best_match_score = score
                        matched_tool = real_tool

                # Replace LLM node data with real tool data if match found
                if matched_tool and best_match_score >= 50:
                    # Keep LLM-generated recommendation_reason and edges, but use real tool data
                    node_data["label"] = matched_tool.get(
                        "Title", node_data.get("label", "")
                    )
                    node_data["description"] = matched_tool.get(
                        "Description", node_data.get("description", "")
                    )
                    node_data["features"] = matched_tool.get(
                        "Features", node_data.get("features", [])
                    )
                    node_data["tags"] = matched_tool.get(
                        "Tags (Keywords)", node_data.get("tags", [])
                    )

                    # Extract official website URL (handle both "Website" and "website" keys)
                    official_website = (
                        matched_tool.get("Website", "")
                        or matched_tool.get("website", "")
                        or ""
                    ).strip()

                    # Check if this is an Internet Search tool
                    tool_source = matched_tool.get("Source", "")
                    is_internet_search = "Internet Search" in tool_source

                    # Validate and use official website URL
                    # For Internet Search tools, accept ANY URL format (including redirect URLs)
                    if official_website:
                        if is_internet_search:
                            # ALWAYS accept Internet Search tool URLs without validation
                            node_data["website"] = official_website
                            logger.info(
                                f"✅ [INTERNET SEARCH] Using website for '{matched_tool.get('Title', 'Unknown')}': {official_website[:100]}..."
                            )
                        elif official_website.startswith(
                            "http://"
                        ) or official_website.startswith("https://"):
                            # For non-Internet Search tools, validate URL format
                            node_data["website"] = official_website
                            logger.info(
                                f"✅ Using official website for '{matched_tool.get('Title', 'Unknown')}': {official_website}"
                            )
                        else:
                            # If no valid website found for non-Internet Search tools, clear it
                            node_data["website"] = ""
                            logger.warning(
                                f"⚠️ No valid official website found for '{matched_tool.get('Title', 'Unknown')}'"
                            )
                    else:
                        node_data["website"] = ""

                    node_data["twitter"] = matched_tool.get("Twitter", "")
                    node_data["facebook"] = matched_tool.get("Facebook", "")
                    node_data["linkedin"] = matched_tool.get("LinkedIn", "")
                    node_data["instagram"] = matched_tool.get("Instagram", "")
                    node_data["source"] = matched_tool.get("Source", "Unknown")
                    print(
                        f"✅ Mapped LLM node '{llm_label}' to real tool: {matched_tool.get('Title', 'Unknown')} "
                        f"(Website: {node_data.get('website', 'N/A')})"
                    )
                else:
                    # No match found - keep LLM node but clear invalid website field
                    print(
                        f"⚠️ No match found for LLM node '{llm_label}' - keeping LLM-generated data"
                    )
                    # If website is not a valid URL (contains tool name or example.com), clear it
                    website = node_data.get("website", "")
                    if (
                        not website
                        or "example.com" in website
                        or not website.startswith("http")
                        or website == llm_label  # Website is same as tool name
                    ):
                        print(
                            f"   ⚠️ Warning: Node has invalid URL '{website}' - clearing it"
                        )
                        node_data["website"] = ""  # Clear invalid website

                mapped_nodes.append(llm_node)

            workflow["nodes"] = mapped_nodes
            return workflow

        except Exception as e:
            print(f"Error mapping LLM nodes to real tools: {e}")
            return workflow

    def _remove_duplicate_nodes(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate nodes from workflow based on tool name and website.
        Keeps the first occurrence of each unique tool.

        Args:
            workflow: Workflow dictionary with nodes

        Returns:
            Workflow with duplicate nodes removed
        """
        try:
            nodes = workflow.get("nodes", [])
            if not nodes:
                return workflow

            unique_nodes = []
            seen_identifiers = set()

            for node in nodes:
                node_data = node.get("data", {})
                tool_name = node_data.get("label", "").strip()
                website = node_data.get("website", "").strip()

                # Create identifier for duplicate detection
                identifier = self._create_node_identifier(tool_name, website)

                # Check if duplicate
                if identifier in seen_identifiers:
                    logger.warning(
                        f"🔄 Removing duplicate node: '{tool_name}' (identifier: {identifier})"
                    )
                    continue

                # Add to seen set and unique list
                seen_identifiers.add(identifier)
                unique_nodes.append(node)

            # Update workflow with unique nodes
            workflow["nodes"] = unique_nodes

            # Also remove edges that reference removed duplicate nodes
            if len(unique_nodes) < len(nodes):
                unique_node_ids = {node.get("id") for node in unique_nodes}
                edges = workflow.get("edges", [])
                valid_edges = [
                    edge
                    for edge in edges
                    if edge.get("source") in unique_node_ids
                    and edge.get("target") in unique_node_ids
                ]
                workflow["edges"] = valid_edges
                logger.info(
                    f"✅ Removed {len(nodes) - len(unique_nodes)} duplicate nodes, "
                    f"kept {len(valid_edges)} valid edges"
                )

            return workflow

        except Exception as e:
            logger.error(f"Error removing duplicate nodes: {e}", exc_info=True)
            return workflow

    def _validate_workflow_websites(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean website fields in all workflow nodes.
        Ensures all website fields are valid URLs (starting with http:// or https://) or empty strings.
        Preserves official website URLs from tool data.
        ALWAYS accepts Internet Search tool URLs without validation.

        Args:
            workflow: Workflow dictionary with nodes

        Returns:
            Workflow with validated website fields
        """
        try:
            nodes = workflow.get("nodes", [])
            for node in nodes:
                node_data = node.get("data", {})
                website = (
                    node_data.get("website", "").strip()
                    if node_data.get("website")
                    else ""
                )

                # Check if this is an Internet Search tool
                tool_source = node_data.get("source", "")
                is_internet_search = "Internet Search" in tool_source

                # If website is not a valid URL, clear it (EXCEPT for Internet Search tools)
                if website:
                    if is_internet_search:
                        # ALWAYS accept Internet Search tool URLs without validation
                        logger.info(
                            f"✅ [INTERNET SEARCH] Preserved website for '{node_data.get('label', 'Unknown')}': {website[:100]}..."
                        )
                    elif not (
                        website.startswith("http://") or website.startswith("https://")
                    ):
                        # For non-Internet Search tools, validate URL format
                        logger.warning(
                            f"Invalid website field in node '{node_data.get('label', 'Unknown')}': "
                            f"'{website}' - clearing it"
                        )
                        node_data["website"] = ""
                    else:
                        # Log when we preserve a valid official website
                        logger.debug(
                            f"✅ Preserved official website for '{node_data.get('label', 'Unknown')}': {website}"
                        )

            return workflow
        except Exception as e:
            logger.error(f"Error validating workflow websites: {e}")
            return workflow

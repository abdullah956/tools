"""Internet search service for finding AI tools using Gemini API with grounding."""

import logging
from typing import Any, Dict, List, Optional

from ai_tool_recommender.ai_agents.core.llm import shared_llm
from ai_tool_recommender.ai_agents.core.validation import ToolDataFormatter
from ai_tool_recommender.ai_agents.tools.internet_search.config import (
    InternetSearchConfig,
)
from ai_tool_recommender.ai_agents.tools.internet_search.gemini_search import (
    GeminiSearchService,
)
from ai_tool_recommender.ai_agents.tools.internet_search.helpers import (
    InternetSearchHelper,
)
from ai_tool_recommender.ai_agents.tools.internet_search.pricing_extractor import (
    PricingExtractor,
)
from ai_tool_recommender.ai_agents.tools.internet_search.tool_extractor import (
    tool_extractor,
)

logger = logging.getLogger(__name__)


class InternetSearchService:
    """Service for searching AI tools on the internet using Gemini API with grounding."""

    def __init__(self):
        """Initialize the internet search service."""
        self.gemini_client = None
        self.config = InternetSearchConfig()
        self.pricing_extractor = PricingExtractor()
        self._initialize_services()

    def _initialize_services(self):
        """Initialize Gemini search client."""
        try:
            logger.info("🔧 [INTERNET SEARCH] Initializing Gemini search service...")
            logger.info(f"🔑 [INTERNET SEARCH] Checking GEMINI_API_KEY configuration...")

            if self.config.is_configured():
                logger.info(
                    f"✅ [INTERNET SEARCH] GEMINI_API_KEY found (length: {len(self.config.gemini_api_key) if self.config.gemini_api_key else 0} chars)"
                )
                self.gemini_client = GeminiSearchService(
                    api_key=self.config.gemini_api_key
                )
                if self.gemini_client.is_initialized():
                    logger.info(
                        "✅ [INTERNET SEARCH] Gemini search client initialized successfully"
                    )
                else:
                    logger.error(
                        "❌ [INTERNET SEARCH] Gemini client failed to initialize"
                    )
            else:
                logger.warning(
                    "⚠️ [INTERNET SEARCH] GEMINI_API_KEY not found in environment variables"
                )
                logger.warning("⚠️ [INTERNET SEARCH] Internet search will be disabled")

        except Exception as e:
            logger.error(
                f"❌ [INTERNET SEARCH] Error initializing internet search services: {e}",
                exc_info=True,
            )

    async def _extract_tool_names_from_query(self, query: str) -> List[str]:
        """Use LLM to extract specific tool names from user query.

        Args:
            query: User's search query

        Returns:
            List of extracted tool names
        """
        try:
            prompt = f"""Extract SPECIFIC tool/software names from this query.
Only return actual tool names, not generic terms.

Query: "{query}"

Rules:
- Return ONLY specific tool names (e.g., "Gusto", "BambooHR", "Slack")
- Do NOT return generic terms (e.g., "payroll tool", "CRM software")
- If no specific tools mentioned, return empty list
- Return as JSON array: ["Tool1", "Tool2"]

Examples:
Query: "I want to use Gusto and BambooHR for payroll"
Response: ["Gusto", "BambooHR"]

Query: "I need a payroll automation tool"
Response: []

Query: "Compare Slack vs Microsoft Teams"
Response: ["Slack", "Microsoft Teams"]

Now extract from the query above:"""

            response = await shared_llm.generate_response(prompt)

            # Parse JSON response
            import json
            import re

            # Extract JSON array from response
            json_match = re.search(r"\[.*?\]", response, re.DOTALL)
            if json_match:
                tools = json.loads(json_match.group())
                logger.info(f"✅ LLM extracted tools: {tools}")
                return tools if isinstance(tools, list) else []

            logger.info("ℹ️ No tools extracted from query")
            return []

        except Exception as e:
            logger.error(f"Error extracting tool names: {e}")
            return []

    def _prepare_search_query(self, query: str, max_length: int = 150) -> str:
        """
        Prepare search query by using LLM to refine if too long.

        Args:
            query: Original query (may be very long)
            max_length: Maximum length for search query

        Returns:
            Refined query suitable for Gemini API (under 150 chars)
        """
        # If query is short enough, return as-is
        if len(query) <= max_length:
            return query

        # Use LLM to intelligently refine the query
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.config.openai_api_key)

            prompt = f"""You are a search query optimizer for finding AI tools on the internet.

Given this detailed project description, extract the MOST IMPORTANT search terms in a concise query (max {max_length} characters).

Focus on:
- Main task/objective
- Key technologies or tools mentioned
- Primary use case

Project Description:
{query}

Return ONLY the refined search query, nothing else. Keep it under {max_length} characters.
Example good outputs:
- "video editing automation Adobe Premiere Pro transitions"
- "AI content generation marketing copy"
- "data analysis automation Python Excel"

Refined search query:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query optimizer. Return only the refined query, no explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )

            refined_query = response.choices[0].message.content.strip()

            # Ensure it's under max_length
            if len(refined_query) > max_length:
                refined_query = refined_query[:max_length].rsplit(" ", 1)[0]

            logger.info(
                f"LLM refined query from {len(query)} to {len(refined_query)} chars: {refined_query}"
            )
            return refined_query

        except Exception as e:
            logger.error(f"Error refining query with LLM: {e}")

            # Fallback: simple truncation with key term extraction
            import re

            # Try to extract from "Project Overview" or "Key Objectives"
            overview_match = re.search(
                r"## Project Overview\s*\n(.*?)(?=\n##|\Z)", query, re.DOTALL
            )
            if overview_match:
                overview = overview_match.group(1).strip()
                result = overview[:max_length].rsplit(" ", 1)[0]
                logger.info(f"Fallback: extracted overview: {result}")
                return result

            # Last resort: just truncate
            result = query[:max_length].rsplit(" ", 1)[0]
            logger.info(f"Fallback: simple truncation: {result}")
            return result

    def _normalize_tool_name(self, name: str) -> str:
        """Normalize tool name for matching.

        Handles:
        - Case insensitivity
        - ".ai" suffix variations (mindhyve.ai -> mindhyve)
        - Common separators

        Args:
            name: Tool name to normalize

        Returns:
            Normalized tool name
        """
        if not name:
            return ""
        normalized = name.lower().strip()
        # Remove common suffixes/prefixes
        normalized = normalized.replace(" - ", " ").replace(" | ", " ")
        # Remove .ai suffix for matching (mindhyve.ai -> mindhyve)
        # But keep it if it's part of the actual name (e.g., "ai.com" should stay)
        if normalized.endswith(".ai") and len(normalized) > 3:
            # Only remove if it's clearly a suffix (not part of domain like "ai.com")
            base = normalized[:-3]
            if base and not base.endswith("."):  # Not like "something.ai.com"
                normalized = base
        return normalized

    def _verify_exact_tool_match(self, result: Dict[str, Any], tool_name: str) -> bool:
        """Verify if search result matches exact tool name.

        Args:
            result: Search result from Gemini
            tool_name: Exact tool name to match

        Returns:
            True if result matches exact tool name, False otherwise
        """
        if not tool_name or not result:
            return False

        title = result.get("title", "").strip()

        if not title:
            return False

        # Normalize for comparison (handles .ai suffix variations)
        normalized_tool = self._normalize_tool_name(tool_name)
        normalized_title = self._normalize_tool_name(title)

        # Also try with .ai suffix if tool_name has it
        tool_with_ai = None
        if tool_name.lower().endswith(".ai") and len(tool_name) > 3:
            tool_with_ai = tool_name.lower().strip()
        elif not tool_name.lower().endswith(".ai"):
            # Try adding .ai for matching
            tool_with_ai = f"{tool_name.lower().strip()}.ai"

        # Check if tool name is in title (exact or partial match)
        # Exact match (normalized)
        if normalized_title == normalized_tool:
            return True

        # Exact match with .ai suffix variations
        if tool_with_ai:
            title_with_ai = normalized_title + ".ai"
            if normalized_title == tool_with_ai or title_with_ai == tool_with_ai:
                return True
            # Also check if title contains tool name with .ai
            if tool_with_ai in normalized_title or normalized_title in tool_with_ai:
                return True

        # Tool name is in title (e.g., "Slack" in "Slack - Team Communication")
        # Additional check: ensure it's not a false positive
        # (e.g., "Slack" shouldn't match "Slackbot" unless tool_name is "Slackbot")
        if (
            normalized_tool in normalized_title
            and len(normalized_tool)
            >= 3  # Lowered from 4 to 3 for shorter names like "Edvenity"
        ):
            # Check if it's at the start of title (more likely to be correct)
            if normalized_title.startswith(normalized_tool):
                return True
            # Check if it's a word boundary match
            import re

            pattern = r"\b" + re.escape(normalized_tool) + r"\b"
            if re.search(pattern, normalized_title):
                return True
                # For shorter names or if word boundary doesn't match, check if tool name is significant part of title
                # (e.g., "Edvenity" should match "Edvenity - AI Interviews")
                title_words = normalized_title.split()
                if normalized_tool in title_words:
                    return True

        # Title is in tool name (e.g., "Microsoft Teams" matches "Teams")
        if (
            normalized_title in normalized_tool and len(normalized_title) >= 3
        ):  # Lowered from 4
            return True

        return False

    def _is_official_site(self, url: str, title: str) -> bool:
        """Check if URL is official product site (not blog/review).

        Args:
            url: URL to check
            title: Page title

        Returns:
            True if URL appears to be official site, False otherwise
        """
        if not url or not url.startswith(("http://", "https://")):
            return False

        url_lower = url.lower()
        title_lower = title.lower()

        # Filter out blog subdomains and review sites
        blog_indicators = [
            "blog.",
            ".blog",
            "medium.com",
            "dev.to",
            "hashnode",
            "wordpress",
            "blogspot",
            "tumblr",
            "ghost",
            "substack",
            "review",
            "comparison",
            "vs",
            "alternatives",
            "best",
            "top",
            "list",
        ]

        for indicator in blog_indicators:
            if indicator in url_lower:
                return False

        # Check for official site indicators
        official_indicators = [
            "pricing",
            "features",
            "product",
            "platform",
            "software",
            "tool",
            "dashboard",
            "login",
            "signup",
            "sign-up",
            "get-started",
            "demo",
            "trial",
        ]

        # URL should have official indicators OR be root domain
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Root domain or main subdomain (www, app, etc.)
            is_main_domain = domain.count(".") <= 2 and not any(
                indicator in domain for indicator in blog_indicators
            )

            # Check URL path for official indicators
            path = parsed.path.lower()
            has_official_path = any(
                indicator in path for indicator in official_indicators
            )

            # Check title for official indicators
            has_official_title = any(
                indicator in title_lower for indicator in official_indicators
            )

            return is_main_domain or has_official_path or has_official_title

        except Exception:
            return False

    async def search_ai_tools_exact(
        self, tool_names: List[str], max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search Internet for exact tool official sites.

        This method searches for specific tool names and verifies that results
        match the exact tool name and are official sites.

        Args:
            tool_names: List of exact tool names to search for
            max_results: Maximum number of results to return per tool

        Returns:
            List of verified AI tool data matching exact tool names
        """
        if (
            not self.gemini_client
            or not self.gemini_client.is_initialized()
            or not tool_names
        ):
            logger.warning("Gemini client not initialized or no tool names provided")
            return []

        try:
            logger.info(
                f"🎯 [INTERNET SEARCH START] Searching Internet for exact tool official sites: {tool_names}"
            )
            all_results = []
            seen_urls = set()

            for tool_name in tool_names:
                # Initialize variables for logging (in case of exceptions)
                gemini_results = []
                scored_results = []
                top_results = []

                try:
                    # Build search query for official site - fully dynamic, no hardcoded assumptions
                    # Handle .ai suffix: try both with and without .ai
                    base_name = tool_name
                    if tool_name.lower().endswith(".ai") and len(tool_name) > 3:
                        base_name = tool_name[:-3].strip()

                    # Build query for Gemini grounding
                    if base_name != tool_name:
                        # Has .ai suffix - search for both versions
                        search_query = f"Find the official website and information about {tool_name} (also known as {base_name}) AI tool or software. Provide official site URL, description, features, and pricing."
                    else:
                        # No .ai suffix - search for the tool name as-is
                        search_query = f"Find the official website and information about {tool_name} AI tool or software. Provide official site URL, description, features, and pricing."

                    logger.info(
                        f"🔍 [INTERNET SEARCH] Searching for '{tool_name}' (base: '{base_name}') official site with Gemini grounding"
                    )
                    logger.info(f"📝 [INTERNET SEARCH] Search query: '{search_query}'")

                    # Perform search with Gemini grounding
                    logger.info(f"🚀 [INTERNET SEARCH] Calling Gemini search API...")
                    search_result = await self.gemini_client.search(
                        query=search_query,
                        max_results=min(
                            max_results * 3, 15
                        ),  # Get more results to find best match
                    )

                    gemini_results = search_result.get("results", [])
                    search_queries_used = search_result.get("search_queries", [])
                    response_text = search_result.get("response_text", "")

                    logger.info(
                        f"✅ [INTERNET SEARCH] Gemini returned {len(gemini_results)} sources for '{tool_name}'"
                    )
                    logger.info(
                        f"🔍 [INTERNET SEARCH] Search queries used by Gemini: {search_queries_used}"
                    )
                    logger.info(
                        f"📄 [INTERNET SEARCH] Response text length: {len(response_text)} chars"
                    )

                    # Log each result
                    for idx, result in enumerate(gemini_results, 1):
                        logger.info(
                            f"  📍 Result #{idx}: {result.get('title', 'No title')} - {result.get('url', 'No URL')}"
                        )

                    if not gemini_results:
                        logger.warning(
                            f"⚠️ Gemini returned NO results for '{tool_name}' with query: {search_query}"
                        )
                        continue

                    # Score and rank results instead of strict filtering
                    scored_results = []
                    processed_count = 0

                    for result in gemini_results:
                        url = result.get("url", "")
                        title = result.get("title", "")

                        # Skip if already processed
                        if url in seen_urls:
                            continue

                        processed_count += 1

                        # Calculate match score (0-100)
                        score = self._calculate_match_score(result, tool_name)

                        # Accept all results (even with low scores) - we'll take the top ones
                        # This ensures we don't miss valid tools due to strict scoring
                        scored_results.append(
                            {
                                "result": result,
                                "score": score,
                                "title": title,
                                "url": url,
                            }
                        )
                        logger.info(f"📊 Scored '{title}': {score:.1f}/100 (URL: {url})")

                    # Sort by score (highest first) and take top 1-3 results
                    scored_results.sort(key=lambda x: x["score"], reverse=True)

                    # Take top results, but ensure we have at least 1 result if any exist
                    # This prevents empty results when scores are low but tools are valid
                    top_results = (
                        scored_results[: min(3, max_results)] if scored_results else []
                    )

                    # If no results have high scores but we have results, take the top 1 anyway
                    if not top_results and scored_results:
                        top_results = [
                            scored_results[0]
                        ]  # Take the best one even if score is low
                        logger.info(
                            f"⚠️ All scores were low, but taking top result anyway: {top_results[0]['title']} (score: {top_results[0]['score']:.1f})"
                        )

                    logger.info(
                        f"✅ Selected top {len(top_results)} results from {len(scored_results)} scored results"
                    )

                    # Process top results
                    for scored_item in top_results:
                        result = scored_item["result"]
                        title = scored_item["title"]
                        url = scored_item["url"]

                        seen_urls.add(url)

                        logger.info(
                            f"✅ Processing best match for '{tool_name}': {title} - {url} (score: {scored_item['score']:.1f})"
                        )

                        # Extract tool data (relaxed validation for best match)
                        tool_data = await self._quick_validate_and_extract_tool(
                            result,
                            tool_name,
                            is_exact_match=False,  # Relaxed - allow best matches
                        )
                        if tool_data:
                            # Use the actual title from result (more accurate) but clean it
                            tool_data["Title"] = ToolDataFormatter._clean_tool_name(
                                title if title else tool_name
                            )
                            tool_data["Source"] = "Internet Search (Best Match)"
                            tool_data["_match_score"] = scored_item["score"]
                            tool_data["_match_type"] = "best_match"
                            all_results.append(tool_data)
                            logger.info(
                                f"✅ Successfully processed tool: {tool_data.get('Title', 'Unknown')} (score: {scored_item['score']:.1f})"
                            )

                            # Stop if we have enough high-quality results
                            if len(all_results) >= max_results:
                                break
                        else:
                            logger.info(
                                f"❌ Tool validation failed for '{title}' - rejected by _quick_validate_and_extract_tool"
                            )

                except Exception as search_error:
                    logger.error(
                        f"❌ Error searching for '{tool_name}': {search_error}",
                        exc_info=True,
                    )
                    continue

                # Log summary for this tool
                if len(gemini_results) > 0:
                    logger.info(
                        f"📊 Tool '{tool_name}' processing summary: "
                        f"Gemini sources: {len(gemini_results)}, "
                        f"Scored: {len(scored_results)}, "
                        f"Top results selected: {len(top_results)}, "
                        f"Accepted: {len([r for r in all_results if r.get('_match_type') == 'best_match'])}"
                    )
                elif not gemini_results:
                    logger.warning(
                        f"⚠️ Tool '{tool_name}': Gemini returned 0 results - tool may not exist or search query needs adjustment"
                    )

            logger.info(
                f"✅ [INTERNET SEARCH END] Internet exact search completed. Found {len(all_results)} exact tool matches out of {sum(len(search_result.get('results', [])) for _ in tool_names)} Gemini sources"
            )

            # Enhance tools with pricing information
            enhanced_results = []
            for tool in all_results[: max_results * len(tool_names)]:
                enhanced_tool = await self.pricing_extractor.enhance_tool_with_pricing(
                    tool
                )
                enhanced_results.append(enhanced_tool)

            return enhanced_results

        except Exception as e:
            logger.error(f"Internet exact search error: {e}")
            return []

    async def search_ai_tools(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for AI tools on the internet using Gemini with grounding.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of verified AI tool data
        """
        print("=" * 100)
        print(f"🌐🌐🌐 GEMINI INTERNET SEARCH CALLED 🌐🌐🌐")
        print(f"Query: '{query}'")
        print(f"Max results: {max_results}")
        print("=" * 100)

        logger.info("=" * 100)
        logger.info(
            "🌐 [GEMINI INTERNET SEARCH] ========================================"
        )
        logger.info(f"🌐 [GEMINI INTERNET SEARCH] STARTING GEMINI WEB SEARCH")
        logger.info(f"📝 [GEMINI INTERNET SEARCH] Query: '{query}'")
        logger.info(f"📊 [GEMINI INTERNET SEARCH] Max results: {max_results}")
        logger.info(
            f"⏱️ [GEMINI INTERNET SEARCH] Start time: {__import__('datetime').datetime.now().isoformat()}"
        )
        logger.info("=" * 100)

        if not self.gemini_client or not self.gemini_client.is_initialized():
            print("❌❌❌ GEMINI CLIENT NOT INITIALIZED ❌❌❌")
            logger.error("=" * 100)
            logger.error("❌ [GEMINI ERROR] Gemini client not initialized!")
            logger.error(
                "❌ [GEMINI ERROR] Check GEMINI_API_KEY in environment variables"
            )
            logger.error("=" * 100)
            return []

        try:
            print(f"✅ Gemini client is initialized, proceeding with search...")
            logger.info(
                "✅ [GEMINI INTERNET SEARCH] Gemini client is initialized and ready"
            )
            logger.info(
                f"🔍 [GEMINI INTERNET SEARCH] Searching internet for AI tools with query: '{query}'"
            )

            # Step 1: Use LLM to extract tool requirements and generate focused searches
            logger.info("🎯 Step 1: Extracting tool requirements using LLM...")
            tool_requirements = await tool_extractor.extract_tool_requirements(query)

            logger.info(f"✅ Tool requirements extracted:")
            logger.info(
                f"   - Has specific tools: {tool_requirements.get('has_specific_tools', False)}"
            )
            logger.info(f"   - Tool names: {tool_requirements.get('tool_names', [])}")
            logger.info(f"   - Tool types: {tool_requirements.get('tool_types', [])}")
            logger.info(
                f"   - Search queries: {tool_requirements.get('search_queries', [])}"
            )

            # Use LLM-generated search queries
            search_queries = tool_requirements.get("search_queries", [])

            if not search_queries:
                # Fallback if LLM didn't generate queries - use the original query directly
                logger.warning(
                    "⚠️ No search queries generated by LLM, using original query as fallback"
                )
                search_queries = [
                    query
                ]  # Use the original query - Gemini will handle it intelligently
                logger.info(f"📝 [FALLBACK] Using query: '{query}'")

            all_results = []
            seen_urls = set()

            for search_query in search_queries:
                try:
                    logger.info(
                        f"🔍 [INTERNET SEARCH] Executing search query: '{search_query}'"
                    )
                    logger.info(f"🚀 [INTERNET SEARCH] Calling Gemini search API...")

                    # Perform search with Gemini grounding
                    search_result = await self.gemini_client.search(
                        query=search_query,
                        max_results=max_results,  # Get full max_results from Gemini
                    )

                    gemini_results = search_result.get("results", [])
                    search_queries_used = search_result.get("search_queries", [])
                    response_text = search_result.get("response_text", "")

                    logger.info(
                        f"✅ [INTERNET SEARCH] Gemini search returned {len(gemini_results)} sources"
                    )
                    logger.info(
                        f"🔍 [INTERNET SEARCH] Search queries used by Gemini: {search_queries_used}"
                    )
                    logger.info(
                        f"📄 [INTERNET SEARCH] Response text length: {len(response_text)} chars"
                    )

                    # Log each result from Gemini
                    logger.info(
                        f"📊 [GEMINI RESULTS] Total results from Gemini: {len(gemini_results)}"
                    )
                    for idx, result in enumerate(gemini_results, 1):
                        logger.info(
                            f"  📍 [GEMINI] Result #{idx}: {result.get('title', 'No title')} - {result.get('url', 'No URL')}"
                        )

                    if not gemini_results:
                        logger.warning(
                            f"⚠️ [GEMINI] No results returned from Gemini for query: '{search_query}'"
                        )
                        logger.warning(f"⚠️ [GEMINI] This could mean:")
                        logger.warning(f"   1. Gemini didn't find any relevant tools")
                        logger.warning(f"   2. The query needs to be more specific")
                        logger.warning(
                            f"   3. There was an API error (check logs above)"
                        )
                        continue

                    # Process results
                    processed_count = 0
                    accepted_count = 0
                    rejected_count = 0

                    for result in gemini_results:
                        processed_count += 1
                        url = result.get("url", "")

                        # Skip if we've already processed this URL
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                        # AGGRESSIVE filtering - ONLY official tool sites, NO blogs/lists/reviews
                        title_lower = result.get("title", "").lower()
                        url_lower = url.lower()

                        # Filter out blogs, lists, reviews, comparisons
                        skip_keywords = [
                            "best",
                            "top",
                            "list",
                            "review",
                            "comparison",
                            "vs",
                            "alternative",
                            "guide",
                            "tutorial",
                            "how to",
                            "what is",
                            "blog",
                            "article",
                            "post",
                            "news",
                            "roundup",
                            "ranking",
                        ]

                        should_skip = False
                        for keyword in skip_keywords:
                            if keyword in title_lower or keyword in url_lower:
                                should_skip = True
                                logger.info(
                                    f"❌ Skipping (contains '{keyword}'): {result.get('title', '')} - {url}"
                                )
                                break

                        if should_skip:
                            rejected_count += 1
                            logger.info(
                                f"❌ [FILTER] Rejected #{processed_count} (contains skip keyword): {result.get('title', '')} - {url}"
                            )
                            continue

                        # Also check with existing helper
                        if InternetSearchHelper.is_blog_or_list(
                            result,
                            self.config.blog_keywords,
                            self.config.blog_domains,
                            self.config.blog_url_patterns,
                        ):
                            rejected_count += 1
                            logger.info(
                                f"❌ [FILTER] Rejected #{processed_count} (blog/list): {result.get('title', '')} - {url}"
                            )
                            continue

                        # RELAXED: Accept results that don't look like blogs/lists/reviews
                        # We trust Gemini's grounding to return relevant tool sites
                        # Only reject if it's clearly a blog/list/review (already filtered above)

                        logger.info(
                            f"✅ [ACCEPT] Processing result #{processed_count}: {result.get('title', '')} - {url}"
                        )

                        # Quick validation - skip heavy verification for speed
                        tool_data = await self._quick_validate_and_extract_tool(
                            result, query
                        )
                        if tool_data:
                            accepted_count += 1
                            all_results.append(tool_data)
                            logger.info(
                                f"✅ [SUCCESS] Successfully processed tool #{accepted_count}: {tool_data.get('Title', 'Unknown')}"
                            )
                        else:
                            rejected_count += 1
                            logger.warning(
                                f"❌ [VALIDATION] Tool validation failed for: {result.get('title', '')} - {url}"
                            )

                        # Stop if we have enough results
                        if len(all_results) >= max_results:
                            break

                    # Log summary for this search query
                    logger.info("=" * 80)
                    logger.info(f"📊 [SEARCH QUERY SUMMARY] Query: '{search_query}'")
                    logger.info(f"   - Gemini results: {len(gemini_results)}")
                    logger.info(f"   - Processed: {processed_count}")
                    logger.info(f"   - Accepted: {accepted_count}")
                    logger.info(f"   - Rejected: {rejected_count}")
                    logger.info(f"   - Total tools so far: {len(all_results)}")
                    logger.info("=" * 80)

                    if len(all_results) >= max_results:
                        logger.info(
                            f"✅ [STOP] Reached max_results ({max_results}), stopping search"
                        )
                        break

                except Exception as search_error:
                    logger.error(
                        f"Error in search query '{search_query}': {search_error}"
                    )
                    continue

            logger.info("=" * 100)
            logger.info(
                f"✅ [GEMINI INTERNET SEARCH] Internet search completed. Found {len(all_results)} AI tools"
            )
            logger.info(
                f"⏱️ [GEMINI INTERNET SEARCH] End time: {__import__('datetime').datetime.now().isoformat()}"
            )
            logger.info("=" * 100)

            # Enhance tools with pricing information if missing
            logger.info(
                f"💰 [GEMINI INTERNET SEARCH] Enhancing {len(all_results)} tools with pricing information..."
            )
            enhanced_results = []
            for idx, tool in enumerate(all_results[:max_results], 1):
                logger.info(
                    f"  💰 Enhancing tool #{idx}: {tool.get('Title', 'Unknown')}"
                )
                enhanced_tool = await self.pricing_extractor.enhance_tool_with_pricing(
                    tool
                )
                enhanced_results.append(enhanced_tool)

            logger.info("=" * 100)
            logger.info(
                f"✅ [GEMINI INTERNET SEARCH] Enhanced {len(enhanced_results)} tools with pricing information"
            )
            logger.info(
                f"📦 [GEMINI INTERNET SEARCH] Returning {len(enhanced_results)} final tools"
            )
            logger.info("=" * 100)
            print("=" * 100)
            print(
                f"✅ GEMINI INTERNET SEARCH COMPLETED: {len(enhanced_results)} tools found"
            )
            print("=" * 100)
            return enhanced_results

        except Exception as e:
            logger.error("=" * 100)
            logger.error("❌ [GEMINI ERROR] ===== GEMINI INTERNET SEARCH FAILED =====")
            logger.error(f"❌ [GEMINI ERROR] Error: {str(e)}")
            logger.error(f"❌ [GEMINI ERROR] Error type: {type(e).__name__}")
            import traceback

            logger.error(f"❌ [GEMINI ERROR] Full traceback:\n{traceback.format_exc()}")
            logger.error("=" * 100)
            print("=" * 100)
            print(f"❌ GEMINI INTERNET SEARCH ERROR: {str(e)}")
            print("=" * 100)
            return []

    def _is_blog_or_list(self, result: Dict[str, Any]) -> bool:
        """Check if a search result is a blog post or list article.

        Args:
            result: Search result from Gemini

        Returns:
            True if it appears to be a blog/list, False otherwise
        """
        title = result.get("title", "").lower()
        url = result.get("url", "").lower()

        # Keywords that indicate blog posts or lists
        blog_keywords = [
            "best",
            "top",
            "list",
            "review",
            "comparison",
            "guide",
            "roundup",
            "vs",
            "alternatives",
            "recommendations",
            "ranking",
            "comparison",
            "blog",
            "article",
            "post",
            "tutorial",
            "how to",
            "tips",
        ]

        # Check title for blog keywords
        for keyword in blog_keywords:
            if keyword in title:
                return True

        # Check URL for blog indicators
        blog_domains = [
            "blog",
            "medium",
            "dev.to",
            "hashnode",
            "wordpress",
            "blogspot",
            "tumblr",
            "ghost",
            "substack",
            "newsletter",
        ]

        return any(domain in url for domain in blog_domains)

    async def _quick_validate_and_extract_tool(
        self, result: Dict[str, Any], query: str, is_exact_match: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Quick validation and extraction without heavy web scraping.

        Args:
            result: Search result from Gemini
            query: Search query or tool name
            is_exact_match: If True, bypasses strict relevance checks (for exact tool name matches)
        """
        try:
            title = result.get("title", "").strip()
            content = result.get("content", "")
            url = result.get("url", "")

            # For exact matches, skip strict relevance checks (we already verified exact match)
            # For best matches (is_exact_match=False), also skip strict checks - let scoring handle it
            if not is_exact_match:
                # Only check for garbage titles - don't do strict relevance check
                title_lower = title.lower()
                garbage_titles = [
                    "untitled",
                    "unknown",
                    "no title",
                    "page not found",
                    "404",
                    "error",
                ]
                if any(garbage in title_lower for garbage in garbage_titles):
                    logger.info(f"❌ Rejecting garbage title: '{title}'")
                    return None
            else:
                # For exact matches, only check for garbage titles
                title_lower = title.lower()
                garbage_titles = [
                    "untitled",
                    "unknown",
                    "no title",
                    "page not found",
                    "404",
                    "error",
                ]
                if any(garbage in title_lower for garbage in garbage_titles):
                    logger.warning(
                        f"❌ Rejecting garbage title for exact match: '{title}'"
                    )
                    return None

            # Final validation before creating tool data - very lenient
            if (
                not title or len(title.strip()) < 2
            ):  # Lowered from 3 to 2 for single-word tool names
                logger.warning(f"❌ Rejecting tool with invalid title: '{title}'")
                return None

            if not url or not url.startswith(("http://", "https://")):
                logger.warning(f"❌ Rejecting tool with invalid URL: '{url}'")
                return None

            # Extract and format tool information
            raw_data = {
                "tool_name": title.strip(),
                "description": (
                    content[:300].strip()
                    if content
                    else f"{title} - AI-powered tool discovered via internet search"
                ),
                "category": await self._generate_tool_category(
                    title.strip(),
                    content[:300].strip() if content else "",
                    query,
                ),
                "features": self._extract_features_from_content(content),
                "website": url,
                "relevance_score": self._calculate_relevance_score(
                    title, content, query
                ),
            }

            tool_data = ToolDataFormatter.format_tool_data(
                raw_data, title, url, query, "Internet Search (Validated)"
            )

            # Generate intelligent recommendation reason
            tool_data[
                "recommendation_reason"
            ] = await self._generate_recommendation_reason(
                tool_data.get("Title", ""),
                tool_data.get("Description", ""),
                tool_data.get("Features", ""),
                query,
            )

            return tool_data

        except Exception as e:
            logger.error(f"Error in quick validation: {e}")
            return None

    def _is_relevant_tool(self, title: str, content: str, query: str) -> bool:
        """Quick relevance check without web scraping."""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()

        # REJECT garbage titles immediately
        garbage_titles = [
            "untitled",
            "unknown",
            "no title",
            "page not found",
            "404",
            "error",
            "home",
            "homepage",
            "welcome",
            "about us",
            "contact",
            "privacy policy",
            "terms of service",
            "blog",
            "news",
            "article",
        ]
        if any(garbage in title_lower for garbage in garbage_titles):
            logger.warning(f"❌ Rejecting garbage title: '{title}'")
            return False

        # Title must be meaningful (at least 3 words or contain specific tool indicators)
        title_words = title_lower.split()
        if len(title_words) < 2:
            logger.warning(f"❌ Rejecting too short title: '{title}'")
            return False

        # Check for AI tool indicators
        ai_indicators = [
            "ai",
            "artificial intelligence",
            "automation",
            "tool",
            "software",
            "platform",
            "app",
            "service",
            "solution",
            "system",
        ]
        has_ai_indicator = any(
            indicator in title_lower or indicator in content_lower
            for indicator in ai_indicators
        )

        # Check for query relevance
        query_words = query_lower.split()
        has_query_relevance = any(
            word in title_lower or word in content_lower
            for word in query_words
            if len(word) > 2
        )

        # Skip blog posts and lists (more aggressive filtering)
        blog_indicators = [
            "best",
            "top",
            "list",
            "review",
            "guide",
            "comparison",
            "vs",
            "alternatives",
            "roundup",
            "collection",
            "ultimate",
            "complete",
            "comprehensive",
            "how to",
            "tips",
            "tricks",
            "tutorial",
            "beginner",
            "advanced",
            "expert",
            "master",
        ]
        is_blog_post = any(indicator in title_lower for indicator in blog_indicators)

        # Must have official product indicators
        product_indicators = [
            "pricing",
            "features",
            "dashboard",
            "login",
            "sign up",
            "free trial",
            "demo",
            "product",
            "solutions",
            "enterprise",
            "business",
            "pro",
            "premium",
        ]
        has_product_indicator = any(
            indicator in title_lower or indicator in content_lower
            for indicator in product_indicators
        )

        is_relevant = (
            has_ai_indicator
            and has_query_relevance
            and not is_blog_post
            and has_product_indicator
        )

        if not is_relevant:
            logger.info(
                f"❌ Rejecting '{title}': ai={has_ai_indicator}, query={has_query_relevance}, blog={is_blog_post}, product={has_product_indicator}"
            )

        return is_relevant

    def _extract_category_from_title(self, title: str) -> str:
        """Extract category from title."""
        title_lower = title.lower()

        if any(word in title_lower for word in ["video", "editing", "editor"]):
            return "Video Editing"
        elif any(word in title_lower for word in ["content", "writing", "copy"]):
            return "Content Creation"
        elif any(word in title_lower for word in ["image", "photo", "graphic"]):
            return "Image Processing"
        elif any(word in title_lower for word in ["automation", "workflow"]):
            return "Automation"
        else:
            return "AI Tool"

    def _extract_meaningful_tags(self, title: str, content: str, query: str) -> str:
        """Extract meaningful tags instead of just using the query."""
        tags = []

        # Extract from title
        title_words = [word.strip() for word in title.lower().split() if len(word) > 2]
        tags.extend(title_words[:3])  # Take first 3 meaningful words from title

        # Add category-based tags
        category = self._extract_category_from_title(title)
        if category != "AI Tool":
            tags.append(category.lower().replace(" ", "_"))

        # Add specific functionality tags from content
        functionality_keywords = [
            "automation",
            "analytics",
            "dashboard",
            "integration",
            "api",
            "workflow",
            "reporting",
            "collaboration",
            "productivity",
            "management",
            "optimization",
        ]
        content_lower = content.lower()
        for keyword in functionality_keywords:
            if keyword in content_lower and keyword not in tags:
                tags.append(keyword)

        # Limit to 5 most relevant tags
        return ", ".join(tags[:5])

    def _extract_features_from_content(self, content: str) -> str:
        """Extract features from content."""
        if not content:
            return "AI-powered features"

        # Simple feature extraction
        features = []
        content_lower = content.lower()

        if "free" in content_lower:
            features.append("Free tier available")
        if "api" in content_lower:
            features.append("API access")
        if "cloud" in content_lower:
            features.append("Cloud-based")
        if "mobile" in content_lower:
            features.append("Mobile support")

        return ", ".join(features) if features else "AI-powered features"

    def _calculate_match_score(self, result: Dict[str, Any], tool_name: str) -> float:
        """Calculate match score (0-100) for a search result.

        Higher score = better match. Uses flexible matching, not exact match.

        Args:
            result: Search result from Gemini
            tool_name: Tool name to match against

        Returns:
            Score from 0-100
        """
        if not result or not tool_name:
            return 0.0

        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "")

        if not title:
            return 0.0

        score = 0.0
        normalized_tool = self._normalize_tool_name(tool_name)
        normalized_title = self._normalize_tool_name(title)
        normalized_url = url.lower()
        normalized_content = content.lower() if content else ""

        # Title matching (most important) - up to 50 points
        if normalized_title == normalized_tool:
            score += 50.0  # Exact match
        elif normalized_tool in normalized_title:
            score += 40.0  # Tool name in title (e.g., "excel" in "Microsoft Excel")
            if normalized_title.startswith(normalized_tool):
                score += 5.0  # Bonus if at start
        elif normalized_title in normalized_tool:
            score += 30.0  # Title in tool name
        else:
            # Check word-level matches
            tool_words = set(normalized_tool.split())
            title_words = set(normalized_title.split())
            common_words = tool_words.intersection(title_words)
            if common_words:
                score += len(common_words) * 10.0  # 10 points per matching word

        # URL matching - up to 20 points
        if normalized_tool in normalized_url:
            score += 20.0
        elif any(
            word in normalized_url for word in normalized_tool.split() if len(word) > 3
        ):
            score += 10.0

        # Official site bonus - up to 20 points
        if self._is_official_site(url, title):
            score += 20.0

        # Content matching - up to 10 points
        if normalized_tool in normalized_content:
            score += 10.0

        # Penalty for blog/review sites
        blog_indicators = ["blog", "review", "article", "news", "comparison"]
        if any(
            indicator in normalized_url or indicator in normalized_title
            for indicator in blog_indicators
        ):
            score -= 15.0

        return max(0.0, min(100.0, score))  # Clamp between 0-100

    def _calculate_relevance_score(self, title: str, content: str, query: str) -> int:
        """Calculate relevance score (1-10)."""
        score = 5  # Base score

        query_lower = query.lower()
        title_lower = title.lower()

        # Boost score for query matches in title
        query_words = query_lower.split()
        title_matches = sum(1 for word in query_words if word in title_lower)
        score += min(title_matches, 3)

        # Boost score for AI indicators
        if "ai" in title_lower or "artificial intelligence" in title_lower:
            score += 2

        return min(score, 10)

    async def _verify_and_extract_tool(
        self, result: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Verify a link by visiting it and extract tool data if it matches the query.

        Args:
            result: Search result from Gemini
            query: Original search query

        Returns:
            Structured tool data or None if verification fails
        """
        try:
            url = result.get("url", "")
            if not url:
                return None

            logger.info(f"Verifying link: {url}")

            # Visit the link and get page content
            page_content = await self._fetch_page_content(url)
            if not page_content:
                logger.warning(f"Could not fetch content from {url}")
                return None

            # Use LLM to verify if this is a relevant AI tool
            verification_result = await self._verify_tool_relevance(
                result, page_content, query
            )

            if not verification_result.get("is_relevant", False):
                logger.info(f"Tool not relevant to query: {result.get('title', '')}")
                return None

            # Extract structured data from verified tool
            tool_data = verification_result.get("tool_data")
            if tool_data:
                logger.info(
                    f"Successfully verified tool: {tool_data.get('Title', 'Unknown')}"
                )
                return tool_data

            return None

        except Exception as e:
            logger.error(f"Error verifying tool: {e}")
            return None

    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch page content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Page content or None if fetch fails
        """
        return await InternetSearchHelper.fetch_page_content(url)

    async def _verify_tool_relevance(
        self, result: Dict[str, Any], page_content: str, query: str
    ) -> Dict[str, Any]:
        """Use LLM to verify if a tool is relevant to the query.

        Args:
            result: Search result from Gemini
            page_content: Content from the actual webpage
            query: Original search query

        Returns:
            Verification result with relevance and tool data
        """
        try:
            title = result.get("title", "")
            url = result.get("url", "")

            verification_prompt = f"""
            Analyze this webpage to determine if it's about a specific AI tool that matches the user's query.

            User Query: "{query}"
            Page Title: "{title}"
            Page URL: "{url}"
            Page Content: "{page_content[:2000]}"

            Determine:
            1. Is this about a SPECIFIC AI TOOL/SOFTWARE (not a blog, list, or article)?
            2. Does it match the user's query needs?
            3. If yes, extract the tool information

            Return ONLY valid JSON:
            {{
                "is_relevant": true/false,
                "relevance_score": 0-10,
                "tool_data": {{
                    "tool_name": "exact tool name (use proper marketing capitalization e.g., HubSpot, DocuSign)",
                    "category": "3-4 word descriptive category (e.g., AI CRM Automation Software)",
                    "description": "what the tool does",
                    "features": "comma-separated key features",
                    "pricing": "pricing info if mentioned",
                    "website": "main website URL",
                    "social_links": {{
                        "twitter": "twitter URL if mentioned",
                        "linkedin": "linkedin URL if mentioned",
                        "facebook": "facebook URL if mentioned",
                        "instagram": "instagram URL if mentioned"
                    }}
                }}
            }}

            If not relevant, set "is_relevant": false and "tool_data": null.
            """

            response_text = await shared_llm.generate_response(verification_prompt)

            # Parse JSON response
            try:
                verification_result = await shared_llm.parse_json_response(
                    response_text
                )
            except Exception as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                return {"is_relevant": False, "tool_data": None}

            # If relevant, format the tool data
            if verification_result.get(
                "is_relevant", False
            ) and verification_result.get("tool_data"):
                tool_data = verification_result["tool_data"]

                # Format tool data using shared formatter
                formatted_tool = ToolDataFormatter.format_tool_data(
                    tool_data, title, url, query, "Internet Search (Verified)"
                )
                # Add relevance score from verification
                formatted_tool["Relevance Score"] = verification_result.get(
                    "relevance_score", 0
                )

                # Generate intelligent recommendation reason
                formatted_tool[
                    "recommendation_reason"
                ] = await self._generate_recommendation_reason(
                    formatted_tool.get("Title", ""),
                    formatted_tool.get("Description", ""),
                    formatted_tool.get("Features", ""),
                    query,
                )

                verification_result["tool_data"] = formatted_tool

            return verification_result

        except Exception as e:
            logger.error(f"Error verifying tool relevance: {e}")
            return {"is_relevant": False, "tool_data": None}

    async def _generate_recommendation_reason(
        self, tool_name: str, description: str, features: str, query: str
    ) -> str:
        """Generate an intelligent recommendation reason based on query and tool capabilities using LLM.

        Args:
            tool_name: Clean tool name
            description: Tool description
            features: Tool features
            query: User's search query

        Returns:
            Meaningful recommendation reason explaining why this tool fits the workflow
        """
        try:
            # Shorten inputs for prompt
            desc_short = description[:300] if description else "N/A"
            input_query = query[:200]

            prompt = f"""Given a user query and a tool, explain WHY this tool is recommended and WHAT it does.
User Query: "{input_query}"
Tool Name: "{tool_name}"
Tool Description: "{desc_short}"

Task: Write a single, concise sentence (max 25 words) explaining why this tool is perfect for this workflow.

CRITICAL RULES:
1. NEVER say "I cannot provide a recommendation", "description is incomplete", or "insufficient information".
2. If description is N/A, use your INTERNAL KNOWLEDGE about "{tool_name}" to explain its value.
3. Start with "Recommended because..." or "Selected to..."
4. Be EXTREMELY POSITIVE and SPECIFIC. Focus on the actual connection between the tool and the goal "{input_query}".
5. Do NOT include any URLs, "Official website" text, or generic marketing fluff.

Reason:"""

            response = await shared_llm.generate_response(prompt)
            reason = response.strip().strip('"')

            # Defensive check
            refusal_keywords = [
                "cannot provide",
                "insufficient",
                "incomplete",
                "not enough information",
            ]
            if reason and any(kw in reason.lower() for kw in refusal_keywords):
                return f"Selected because {tool_name} offers specialized automation features that directly address your requirement for '{input_query}'."

            return reason

        except Exception as e:
            logger.warning(f"Error generating recommendation reason: {e}")
            # Fallback to simple construction
            return f"Selected to support your workflow with {tool_name}'s core capabilities."

    async def _generate_tool_category(
        self, tool_name: str, description: str, query: str
    ) -> str:
        """Generate a concise 3-4 word category for the tool using LLM.

        Args:
            tool_name: Tool name
            description: Tool description
            query: User's search query

        Returns:
            A 3-4 word category (e.g., "AI Content Marketing Platform")
        """
        try:
            prompt = f"""Given an AI tool and a user's workflow query, provide a concise 3-4 word category for the tool.
Tool Name: "{tool_name}"
Tool Description: "{description[:200]}"
User Query: "{query[:200]}"

Task: Return ONLY a 3-4 word category that describes the tool's primary function in this context.
Use proper marketing capitalization for any names included.
Do NOT use more than 4 words.
Do NOT use generic terms like "AI Tool" alone.

Example: "AI Lead Generation Software" or "Automated Sales Reporting Tool"

Category:"""

            response = await shared_llm.generate_response(prompt)
            # Remove quotes and limit to 4 words just in case
            category = response.strip().strip('"')
            words = category.split()
            if len(words) > 4:
                return " ".join(words[:4])
            return category

        except Exception as e:
            logger.warning(f"Error generating tool category: {e}")
            return "AI Powered Tool"

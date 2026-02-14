"""Internet search service for finding AI tools using Tavily API with link verification."""

import logging
from typing import Any, Dict, List, Optional

from tavily import TavilyClient

from microservices.ai_tool_recommender.ai_agents.core.llm import shared_llm
from microservices.ai_tool_recommender.ai_agents.core.validation import (
    ToolDataFormatter,
)
from microservices.ai_tool_recommender.ai_agents.tools.internet_search.config import (
    InternetSearchConfig,
)
from microservices.ai_tool_recommender.ai_agents.tools.internet_search.helpers import (
    InternetSearchHelper,
)
from microservices.ai_tool_recommender.ai_agents.tools.internet_search.pricing_extractor import (
    PricingExtractor,
)

logger = logging.getLogger(__name__)


class InternetSearchService:
    """Service for searching AI tools on the internet using Tavily API with link verification."""

    def __init__(self):
        """Initialize the internet search service."""
        self.tavily_client = None
        self.config = InternetSearchConfig()
        self.pricing_extractor = PricingExtractor()
        self._initialize_services()

    def _initialize_services(self):
        """Initialize Tavily client."""
        try:
            if self.config.is_configured():
                self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
                logger.info("Tavily client initialized successfully")
            else:
                logger.warning("TAVILY_API_KEY not found in environment variables")

        except Exception as e:
            logger.error(f"Error initializing internet search services: {e}")

    async def search_ai_tools(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for AI tools on the internet using Tavily and verify links.

        Args:
            query: The search query
            max_results: Maximum number of results to return

        Returns:
            List of verified AI tool data
        """
        if not self.tavily_client:
            logger.error("Tavily client not initialized")
            return []

        try:
            logger.info(f"Searching internet for AI tools with query: {query}")

            # Create targeted search queries for faster results
            search_queries = [
                f'"{query}" AI tool software -"best" -"top" -"list"',
                f'"{query}" AI software application',
            ]

            all_results = []
            seen_urls = set()

            for search_query in search_queries:
                try:
                    logger.info(f"Executing search query: {search_query}")

                    # Perform search with Tavily (optimized for speed)
                    search_result = self.tavily_client.search(
                        query=search_query,
                        search_depth="basic",
                        max_results=min(max_results, 5),  # Limit results for speed
                    )

                    logger.info(
                        f"Tavily search returned {len(search_result.get('results', []))} results"
                    )

                    # Process results
                    for result in search_result.get("results", []):
                        url = result.get("url", "")

                        # Skip if we've already processed this URL
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)

                        # Skip blog posts and list articles (less aggressive filtering)
                        if InternetSearchHelper.is_blog_or_list(
                            result, self.config.blog_keywords, self.config.blog_domains
                        ):
                            logger.info(
                                f"Skipping blog/list result: {result.get('title', '')} - {result.get('url', '')}"
                            )
                            continue

                        logger.info(
                            f"Processing result: {result.get('title', '')} - {result.get('url', '')}"
                        )

                        # Quick validation - skip heavy verification for speed
                        tool_data = await self._quick_validate_and_extract_tool(
                            result, query
                        )
                        if tool_data:
                            all_results.append(tool_data)
                            logger.info(
                                f"Successfully processed tool: {tool_data.get('Title', 'Unknown')}"
                            )

                        # Stop if we have enough results
                        if len(all_results) >= max_results:
                            break

                    if len(all_results) >= max_results:
                        break

                except Exception as search_error:
                    logger.error(
                        f"Error in search query '{search_query}': {search_error}"
                    )
                    continue

            logger.info(f"Internet search completed. Found {len(all_results)} AI tools")

            # Enhance tools with pricing information if missing
            enhanced_results = []
            for tool in all_results[:max_results]:
                enhanced_tool = await self.pricing_extractor.enhance_tool_with_pricing(
                    tool
                )
                enhanced_results.append(enhanced_tool)

            logger.info(
                f"Enhanced {len(enhanced_results)} tools with pricing information"
            )
            return enhanced_results

        except Exception as e:
            logger.error(f"Internet search error: {e}")
            return []

    def _is_blog_or_list(self, result: Dict[str, Any]) -> bool:
        """Check if a search result is a blog post or list article.

        Args:
            result: Search result from Tavily

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
        self, result: Dict[str, Any], query: str
    ) -> Optional[Dict[str, Any]]:
        """Quick validation and extraction without heavy web scraping."""
        try:
            title = result.get("title", "").strip()
            content = result.get("content", "")
            url = result.get("url", "")

            # Quick relevance check using title and content
            if not self._is_relevant_tool(title, content, query):
                return None

            # Extract basic information without web scraping
            tool_data = {
                "Title": title,
                "Description": (
                    content[:300] if content else "AI tool found via internet search"
                ),
                "Category": self._extract_category_from_title(title),
                "Features": self._extract_features_from_content(content),
                "Tags (Keywords)": query,
                "Website": url,
                "Twitter": "",
                "Facebook": "",
                "Linkedin": "",
                "Instagram": "",
                "Price From": "",
                "Price To": "",
                "Source": "Internet Search (Quick Validated)",
                "Relevance Score": self._calculate_relevance_score(
                    title, content, query
                ),
            }

            return tool_data

        except Exception as e:
            logger.error(f"Error in quick validation: {e}")
            return None

    def _is_relevant_tool(self, title: str, content: str, query: str) -> bool:
        """Quick relevance check without web scraping."""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()

        # Check for AI tool indicators
        ai_indicators = [
            "ai",
            "artificial intelligence",
            "automation",
            "tool",
            "software",
            "platform",
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

        # Skip blog posts and lists
        blog_indicators = [
            "best",
            "top",
            "list",
            "review",
            "guide",
            "comparison",
            "vs",
            "alternatives",
        ]
        is_blog_post = any(indicator in title_lower for indicator in blog_indicators)

        return has_ai_indicator and has_query_relevance and not is_blog_post

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
            result: Search result from Tavily
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
            result: Search result from Tavily
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
                    "tool_name": "exact tool name",
                    "category": "tool category",
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

                verification_result["tool_data"] = formatted_tool

            return verification_result

        except Exception as e:
            logger.error(f"Error verifying tool relevance: {e}")
            return {"is_relevant": False, "tool_data": None}

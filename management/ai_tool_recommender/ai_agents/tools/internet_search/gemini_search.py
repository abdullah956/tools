"""Gemini Web Search service for finding AI tools using Google's Gemini API with grounding."""

import logging
from typing import Any, Dict, List

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiSearchService:
    """Service for searching the web using Gemini's grounding feature."""

    def __init__(self, api_key: str):
        """Initialize Gemini search service.

        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Gemini client."""
        try:
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("✅ Gemini client initialized successfully")
            else:
                logger.warning("⚠️ GEMINI_API_KEY not provided")
        except Exception as e:
            logger.error(f"❌ Error initializing Gemini client: {e}")
            self.client = None

    async def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform web search using Gemini's grounding feature.

        Args:
            query: Search query
            max_results: Maximum number of results (note: Gemini controls actual count)

        Returns:
            Dictionary with search results and metadata
        """
        if not self.client:
            logger.error("❌ Gemini client not initialized")
            return {"results": [], "metadata": {}}

        try:
            logger.info(
                f"🔍 [GEMINI SEARCH START] Query: '{query}' | Max results: {max_results}"
            )
            logger.info(
                f"📝 [GEMINI] Sending request to Gemini API with grounding enabled..."
            )

            # Create grounding tool that uses Google Search
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            logger.info("✅ [GEMINI] Grounding tool (Google Search) configured")

            # Configure the request to use the search tool
            config = types.GenerateContentConfig(tools=[grounding_tool])
            logger.info("✅ [GEMINI] Request config created with grounding tool")

            # Prepare the prompt - optimized for tool discovery
            prompt = f"""Find AI tools and software for: {query}

Search for official websites and product pages of AI tools, software platforms, and applications that match this query.

Requirements:
- Return official product websites (not blog posts, reviews, or comparison articles)
- Focus on actual tools and software platforms
- Include tools for: {query}

Return information about multiple relevant tools if available. For each tool, provide:
1. Official website URL
2. Tool name and brief description
3. Key features or capabilities

Focus on finding real, active AI tools and software platforms."""

            logger.info(
                f"📤 [GEMINI] Calling generate_content with model: gemini-2.5-flash"
            )
            logger.info(f"📤 [GEMINI] Prompt length: {len(prompt)} characters")

            # Generate content with grounding
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",  # Using gemini-2.5-flash as per user example
                contents=prompt,
                config=config,
            )

            logger.info("✅ [GEMINI] Received response from Gemini API")
            logger.info(
                f"📊 [GEMINI] Response has {len(response.candidates) if response.candidates else 0} candidates"
            )

            # Extract response text
            response_text = (
                response.text if hasattr(response, "text") and response.text else ""
            )
            logger.info(
                f"📄 [GEMINI] Response text length: {len(response_text)} characters"
            )
            if response_text:
                logger.info(f"📄 [GEMINI] Response preview: {response_text[:200]}...")

            # Extract grounding metadata
            results = []
            search_queries = []
            sources = []

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                logger.info("✅ [GEMINI] Processing candidate for grounding metadata...")

                # Extract grounding metadata if available
                if (
                    hasattr(candidate, "grounding_metadata")
                    and candidate.grounding_metadata
                ):
                    metadata = candidate.grounding_metadata
                    logger.info("✅ [GEMINI] Grounding metadata found!")

                    # Extract search queries used
                    if (
                        hasattr(metadata, "web_search_queries")
                        and metadata.web_search_queries
                    ):
                        search_queries = list(metadata.web_search_queries)
                        logger.info(
                            f"🔍 [GEMINI] Web search queries used by Gemini: {search_queries}"
                        )
                    else:
                        logger.warning(
                            "⚠️ [GEMINI] No web_search_queries found in grounding metadata"
                        )

                    # Extract grounding chunks (sources)
                    if (
                        hasattr(metadata, "grounding_chunks")
                        and metadata.grounding_chunks
                    ):
                        logger.info(
                            f"📚 [GEMINI] Found {len(metadata.grounding_chunks)} grounding chunks"
                        )
                        for idx, chunk in enumerate(metadata.grounding_chunks):
                            if hasattr(chunk, "web") and chunk.web:
                                try:
                                    url = (
                                        chunk.web.uri
                                        if hasattr(chunk.web, "uri")
                                        else ""
                                    )
                                    title = (
                                        chunk.web.title
                                        if hasattr(chunk.web, "title")
                                        else ""
                                    )
                                    if url:  # Only add if we have a URL
                                        # Extract domain name from URL for unique content
                                        domain = (
                                            url.split("//")[-1]
                                            .split("/")[0]
                                            .replace("www.", "")
                                        )

                                        # Create unique content for each tool based on title and URL
                                        # This prevents deduplication and ensures each tool is treated separately
                                        unique_content = f"{title}\n\nOfficial website: {url}\n\nDomain: {domain}\n\n{response_text[:500]}"

                                        source = {
                                            "url": url,
                                            "title": title,
                                            "content": unique_content,  # Unique content per tool
                                        }
                                        sources.append(source)
                                        results.append(source)
                                        logger.info(
                                            f"  ✅ [GEMINI] Source #{idx+1}: {title} - {url}"
                                        )
                                    else:
                                        logger.warning(
                                            f"  ⚠️ [GEMINI] Source #{idx+1}: Missing URL, skipping"
                                        )
                                except Exception as e:
                                    logger.warning(
                                        f"  ❌ [GEMINI] Error extracting source #{idx+1}: {e}"
                                    )
                                    continue
                    else:
                        logger.warning(
                            "⚠️ [GEMINI] No grounding_chunks found in metadata"
                        )
                else:
                    logger.warning(
                        "⚠️ [GEMINI] No grounding_metadata found in candidate"
                    )
            else:
                logger.warning("⚠️ [GEMINI] No candidates in response")

            logger.info(
                f"✅ [GEMINI SEARCH END] Completed: {len(results)} sources extracted, {len(search_queries)} search queries used"
            )
            logger.info(f"📊 [GEMINI] Results summary:")
            for idx, result in enumerate(results[:max_results], 1):
                logger.info(
                    f"  {idx}. {result.get('title', 'No title')} - {result.get('url', 'No URL')}"
                )

            # Return all results (don't limit here - let caller decide)
            logger.info(
                f"📦 [GEMINI] Returning {len(results)} results (requested: {max_results})"
            )
            return {
                "results": results,  # Return all results, don't limit
                "response_text": response.text if response.text else "",
                "search_queries": search_queries,
                "total_sources": len(sources),
            }

        except Exception as e:
            logger.error(f"❌ Error in Gemini search: {e}", exc_info=True)
            return {"results": [], "metadata": {}, "error": str(e)}

    async def search_with_specific_query(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform targeted search for specific tool information.

        Args:
            query: Specific search query (e.g., "Slack official website features")
            max_results: Maximum number of results

        Returns:
            List of search results with URL, title, and content
        """
        result = await self.search(query, max_results)
        return result.get("results", [])

    def is_initialized(self) -> bool:
        """Check if Gemini client is initialized.

        Returns:
            True if client is ready, False otherwise
        """
        return self.client is not None

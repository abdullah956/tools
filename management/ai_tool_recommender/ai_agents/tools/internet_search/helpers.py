"""Internet search utilities and helpers."""

import logging
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class InternetSearchHelper:
    """Helper class for Internet search operations."""

    @staticmethod
    async def fetch_page_content(url: str, timeout: int = 10) -> Optional[str]:
        """Fetch page content from a URL.

        Args:
            url: URL to fetch
            timeout: Request timeout in seconds

        Returns:
            Page content or None if fetch fails
        """
        try:
            # Skip Gemini redirect URLs - they return 403 and aren't meant to be accessed directly
            if "vertexaisearch.cloud.google.com/grounding-api-redirect" in url:
                logger.warning(
                    f"⚠️ Skipping Gemini redirect URL (not accessible): {url[:100]}..."
                )
                return None

            timeout_config = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Parse HTML and extract text content
                        soup = BeautifulSoup(html, "html.parser")

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Get text content
                        text = soup.get_text()

                        # Clean up text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (
                            phrase.strip()
                            for line in lines
                            for phrase in line.split("  ")
                        )
                        text = " ".join(chunk for chunk in chunks if chunk)

                        # Limit content length
                        return text[:3000] if len(text) > 3000 else text
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    @staticmethod
    def is_blog_or_list(
        result: Dict[str, Any],
        blog_keywords: List[str],
        blog_domains: List[str],
        blog_url_patterns: List[str] = None,
    ) -> bool:
        """Check if a search result is a blog post or list article.

        Args:
            result: Search result from Gemini
            blog_keywords: List of keywords that indicate blogs/lists
            blog_domains: List of domains that indicate blogs
            blog_url_patterns: List of URL patterns that indicate blogs (e.g., /blog/, /article/)

        Returns:
            True if it appears to be a blog/list, False otherwise
        """
        title = result.get("title", "").lower()
        url = result.get("url", "").lower()

        # Check title for blog keywords
        if any(keyword in title for keyword in blog_keywords):
            return True

        # Check URL for blog domains
        if any(domain in url for domain in blog_domains):
            return True

        # Check URL for blog patterns (e.g., /blog/, /article/)
        if blog_url_patterns and any(pattern in url for pattern in blog_url_patterns):
            return True

        return False

    @staticmethod
    def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL.

        Args:
            results: List of search results

        Returns:
            Deduplicated list of results
        """
        seen_urls = set()
        unique_results = []

        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        logger.info(f"Deduplicated {len(results)} results to {len(unique_results)}")
        return unique_results

    @staticmethod
    def format_tool_data(
        tool_data: Dict[str, Any], title: str, url: str, query: str
    ) -> Dict[str, Any]:
        """Format extracted tool data into standard format.

        Args:
            tool_data: Raw tool data from LLM
            title: Page title
            url: Page URL
            query: Original search query

        Returns:
            Formatted tool data
        """
        return {
            "Title": tool_data.get("tool_name", title),
            "Description": tool_data.get("description", ""),
            "Category": tool_data.get("category", "AI Tool"),
            "Features": tool_data.get("features", ""),
            "Tags (Keywords)": query,
            "Website": tool_data.get("website", url),
            "Twitter": tool_data.get("social_links", {}).get("twitter", ""),
            "Facebook": tool_data.get("social_links", {}).get("facebook", ""),
            "Linkedin": tool_data.get("social_links", {}).get("linkedin", ""),
            "Instagram": tool_data.get("social_links", {}).get("instagram", ""),
            "Price From": tool_data.get("pricing", ""),
            "Price To": "",
            "Source": "Internet Search (Verified)",
            "Relevance Score": tool_data.get("relevance_score", 0),
        }

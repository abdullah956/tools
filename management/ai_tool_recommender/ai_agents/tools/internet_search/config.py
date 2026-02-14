"""Internet search configuration and utilities."""

import os
from typing import Any, Dict, List


class InternetSearchConfig:
    """Configuration for Internet Search service."""

    def __init__(self):
        """Initialize Internet Search configuration."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.max_results_per_query = int(
            os.getenv("INTERNET_MAX_RESULTS_PER_QUERY", "5")
        )
        self.max_total_results = int(os.getenv("INTERNET_MAX_TOTAL_RESULTS", "10"))
        self.search_depth = os.getenv("INTERNET_SEARCH_DEPTH", "basic")
        self.timeout_seconds = int(os.getenv("INTERNET_TIMEOUT_SECONDS", "10"))

        # Blog filtering keywords (AGGRESSIVE - only official tool sites)
        self.blog_keywords = [
            "best ai tools",
            "top ai tools",
            "list of ai tools",
            "ai tools review",
            "ai tools comparison",
            "ai tools guide",
            "ai tools roundup",
            "ai tools vs",
            "ai tools alternatives",
            "ai tools recommendations",
            "ai tools ranking",
            "blog post",
            "article about",
            "tutorial on",
            "how to choose",
            "tips for choosing",
            "how to",
            "what is",
            "saves time",
            "reduces errors",
            "benefits of",
            "why use",
            "guide to",
            "introduction to",
        ]

        # Blog URL patterns to filter out (AGGRESSIVE)
        self.blog_url_patterns = [
            "/blog/",
            "/article/",
            "/post/",
            "/news/",
            "/learn/",
            "/resources/",
            "/guide/",
            "/tutorial/",
            "/insights/",
            "/knowledge/",
            "/docs/",
            "/documentation/",
            "/developer/",
            "/developers/",
            "/api/",
        ]

        # Blog domains to filter out
        self.blog_domains = [
            "medium.com",
            "dev.to",
            "hashnode.com",
            "wordpress.com",
            "blogspot.com",
            "tumblr.com",
            "ghost.io",
            "substack.com",
            "forbes.com",
            "techcrunch.com",
            "venturebeat.com",
        ]

    def is_configured(self) -> bool:
        """Check if Internet Search is properly configured."""
        return bool(self.gemini_api_key)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return {
            "gemini_api_key": self.gemini_api_key,
            "max_results_per_query": self.max_results_per_query,
            "max_total_results": self.max_total_results,
            "search_depth": self.search_depth,
            "timeout_seconds": self.timeout_seconds,
            "blog_keywords": self.blog_keywords,
            "blog_domains": self.blog_domains,
        }


def validate_internet_search_config() -> str:
    """Validate Internet Search configuration and return error message if invalid."""
    config = InternetSearchConfig()

    if not config.is_configured():
        return "GEMINI_API_KEY not found in environment variables"

    if config.max_results_per_query <= 0:
        return "INTERNET_MAX_RESULTS_PER_QUERY must be greater than 0"

    if config.max_total_results <= 0:
        return "INTERNET_MAX_TOTAL_RESULTS must be greater than 0"

    return None


def generate_search_queries(base_query: str) -> List[str]:
    """Generate multiple search queries for better coverage.

    Args:
        base_query: Base search query

    Returns:
        List of search queries
    """
    queries = [
        f'"{base_query}" AI tool software -"best" -"top" -"list" -"review" -"comparison" -"guide"',
        f'"{base_query}" AI software application pricing features',
        f'"{base_query}" AI tool platform service',
        f'"{base_query}" AI automation software',
    ]

    return queries

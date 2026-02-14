"""Async workers for parallel processing and background tasks."""

import asyncio
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

from microservices.ai_tool_recommender.ai_agents.core.llm import shared_llm

logger = logging.getLogger(__name__)


class AsyncScraper:
    """High-performance async web scraper."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize the async scraper with concurrency control."""
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a single URL.

        Args:
            url: URL to scrape

        Returns:
            Scraped data
        """
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, "html.parser")

                            # Extract basic info
                            title = soup.find("title")
                            title_text = title.get_text().strip() if title else ""

                            # Remove scripts and styles
                            for script in soup(["script", "style"]):
                                script.decompose()

                            text = soup.get_text()
                            clean_text = " ".join(text.split())

                            # Extract meta tags
                            meta_tags = {}
                            for meta in soup.find_all("meta"):
                                name = meta.get("name") or meta.get("property")
                                content = meta.get("content")
                                if name and content:
                                    meta_tags[name] = content

                            # Extract links
                            links = []
                            for link in soup.find_all("a", href=True):
                                href = link.get("href")
                                text = link.get_text().strip()
                                if href and text:
                                    links.append({"url": href, "text": text})

                            return {
                                "url": url,
                                "title": title_text,
                                "content": clean_text[:3000],  # Limit content
                                "meta_tags": meta_tags,
                                "links": links[:20],  # Limit links
                                "status": "success",
                                "scraped_at": datetime.now().isoformat(),
                            }
                        else:
                            return {
                                "url": url,
                                "status": "error",
                                "error": f"HTTP {response.status}",
                                "scraped_at": datetime.now().isoformat(),
                            }
            except Exception as e:
                return {
                    "url": url,
                    "status": "error",
                    "error": str(e),
                    "scraped_at": datetime.now().isoformat(),
                }

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of scraped data
        """
        logger.info(f"Scraping {len(urls)} URLs in parallel")

        tasks = [self.scrape_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    {
                        "url": urls[i],
                        "status": "error",
                        "error": str(result),
                        "scraped_at": datetime.now().isoformat(),
                    }
                )
            else:
                final_results.append(result)

        successful = sum(1 for r in final_results if r.get("status") == "success")
        logger.info(f"Scraping completed: {successful}/{len(urls)} successful")

        return final_results


class LLMExtractor:
    """LLM-based data extractor for structured information."""

    def __init__(self):
        """Initialize the LLM data extractor with extraction prompt."""
        self.extraction_prompt = """
        SYSTEM: You are a data extraction expert. Extract structured information from webpage content.

        WEBPAGE DATA:
        Title: {title}
        URL: {url}
        Content: {content}

        TASK: Extract structured information about AI tools from this webpage.

        Return ONLY valid JSON:
        {{
            "tool_name": "exact tool name",
            "description": "what the tool does",
            "features": ["feature1", "feature2", "feature3"],
            "pricing": ["pricing info if mentioned"],
            "social_links": {{
                "twitter": "twitter URL if found",
                "linkedin": "linkedin URL if found",
                "facebook": "facebook URL if found",
                "instagram": "instagram URL if found"
            }},
            "is_sponsored": false,
            "relevance_score": 0.8,
            "extraction_confidence": 0.9
        }}

        IMPORTANT:
        - Only extract if this is about a specific AI tool
        - Set is_sponsored=true if this looks like an advertisement
        - relevance_score: 0-1 (how relevant to AI tools)
        - extraction_confidence: 0-1 (how confident in extraction)
        - Return null for missing fields
        """

    async def extract_tool_data(
        self, scraped_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract structured tool data from scraped content.

        Args:
            scraped_data: Scraped webpage data

        Returns:
            Extracted tool data or None
        """
        try:
            if scraped_data.get("status") != "success":
                return None

            # Use LLM to extract structured data
            prompt = self.extraction_prompt.format(
                title=scraped_data.get("title", ""),
                url=scraped_data.get("url", ""),
                content=scraped_data.get("content", "")[:2000],  # Limit content
            )

            response = await shared_llm.generate_response(prompt)
            extracted_data = await shared_llm.parse_json_response(response)

            # Validate extraction
            if not extracted_data.get("tool_name"):
                return None

            # Add metadata
            extracted_data.update(
                {
                    "source_url": scraped_data.get("url"),
                    "extracted_at": datetime.now().isoformat(),
                    "scraped_data": scraped_data,
                }
            )

            return extracted_data

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None

    async def extract_from_multiple(
        self, scraped_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract tool data from multiple scraped pages.

        Args:
            scraped_data_list: List of scraped data

        Returns:
            List of extracted tool data
        """
        logger.info(f"Extracting tool data from {len(scraped_data_list)} pages")

        tasks = [self.extract_tool_data(data) for data in scraped_data_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        extracted_tools = []
        for result in results:
            if isinstance(result, dict) and result.get("tool_name"):
                extracted_tools.append(result)

        logger.info(f"Extraction completed: {len(extracted_tools)} tools extracted")
        return extracted_tools


class DeduplicationEngine:
    """Engine for deduplicating and merging tool data."""

    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize the deduplication engine with similarity threshold."""
        self.similarity_threshold = similarity_threshold

    def calculate_similarity(
        self, tool1: Dict[str, Any], tool2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two tools.

        Args:
            tool1: First tool data
            tool2: Second tool data

        Returns:
            Similarity score (0-1)
        """
        try:
            # Simple similarity based on name and description
            name1 = tool1.get("tool_name", "").lower()
            name2 = tool2.get("tool_name", "").lower()

            desc1 = tool1.get("description", "").lower()
            desc2 = tool2.get("description", "").lower()

            # Name similarity
            name_similarity = 1.0 if name1 == name2 else 0.0

            # Description similarity (simple word overlap)
            words1 = set(desc1.split())
            words2 = set(desc2.split())

            if words1 and words2:
                desc_similarity = len(words1.intersection(words2)) / len(
                    words1.union(words2)
                )
            else:
                desc_similarity = 0.0

            # Weighted combination
            total_similarity = (name_similarity * 0.7) + (desc_similarity * 0.3)
            return total_similarity

        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0

    def deduplicate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate tools based on similarity.

        Args:
            tools: List of tool data

        Returns:
            Deduplicated list of tools
        """
        logger.info(f"Deduplicating {len(tools)} tools")

        if not tools:
            return []

        # Sort tools by confidence score
        sorted_tools = sorted(
            tools, key=lambda x: x.get("extraction_confidence", 0), reverse=True
        )

        deduplicated = []
        for tool in sorted_tools:
            is_duplicate = False

            for existing_tool in deduplicated:
                similarity = self.calculate_similarity(tool, existing_tool)

                if similarity >= self.similarity_threshold:
                    # Merge tools (keep the one with higher confidence)
                    if tool.get("extraction_confidence", 0) > existing_tool.get(
                        "extraction_confidence", 0
                    ):
                        # Replace existing tool
                        deduplicated.remove(existing_tool)
                        deduplicated.append(tool)
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(tool)

        logger.info(f"Deduplication completed: {len(deduplicated)} unique tools")
        return deduplicated


class AsyncWorkerPool:
    """Pool of async workers for parallel processing."""

    def __init__(self, max_workers: int = 5):
        """Initialize the async worker pool with max workers."""
        self.max_workers = max_workers
        self.scraper = AsyncScraper(max_concurrent=max_workers)
        self.extractor = LLMExtractor()
        self.deduplicator = DeduplicationEngine()

    async def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process URLs through complete pipeline.

        Args:
            urls: List of URLs to process

        Returns:
            List of processed tool data
        """
        try:
            logger.info(f"Processing {len(urls)} URLs through worker pool")

            # Step 1: Scrape URLs in parallel
            scraped_data = await self.scraper.scrape_urls(urls)

            # Step 2: Extract tool data in parallel
            extracted_tools = await self.extractor.extract_from_multiple(scraped_data)

            # Step 3: Deduplicate tools
            final_tools = self.deduplicator.deduplicate_tools(extracted_tools)

            logger.info(
                f"Worker pool processing completed: {len(final_tools)} final tools"
            )
            return final_tools

        except Exception as e:
            logger.error(f"Worker pool processing error: {e}")
            return []


# Global instances
async_scraper = AsyncScraper()
llm_extractor = LLMExtractor()
deduplication_engine = DeduplicationEngine()
async_worker_pool = AsyncWorkerPool()

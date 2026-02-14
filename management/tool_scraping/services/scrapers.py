"""Apify scraping service and HTML cleaning utilities."""

import logging
import os
from urllib.parse import urljoin, urlparse

import requests
from apify_client import ApifyClient
from bs4 import BeautifulSoup
from defusedxml import ElementTree as ET

logger = logging.getLogger(__name__)


class ApifyService:
    """Service for fetching web pages using Apify."""

    def __init__(self):
        """Initialize ApifyService with API token."""
        self.token = os.environ.get("APIFY_API_KEY")
        if not self.token:
            logger.warning("APIFY_API_KEY not found in environment variables")
            self.client = None
        else:
            self.client = ApifyClient(self.token)

    def fetch_page_content(self, url, return_markdown=True):
        """
        Fetches the raw HTML or markdown of a page using Apify.

        By default, fetches markdown format for better text structure preservation.

        Args:
            url (str): URL to fetch
            return_markdown (bool): If True (default), returns markdown. If False, returns HTML.

        Returns:
            str: Raw HTML or markdown content
        """
        if not self.client:
            logger.error("Apify client not initialized")
            raise ValueError("Apify client not initialized - check APIFY_API_KEY")

        try:
            logger.info(f"Fetching page via Apify: {url} (markdown={return_markdown})")

            if return_markdown:
                # Use website-content-crawler for markdown support
                run_input = {
                    "startUrls": [{"url": url}],
                    "maxCrawlPages": 1,
                    "crawlerType": "playwright:firefox",
                    "saveMarkdown": True,
                }

                # Run the website-content-crawler actor
                run = self.client.actor("apify/website-content-crawler").call(
                    run_input=run_input
                )

                # Fetch results
                dataset_items = (
                    self.client.dataset(run["defaultDatasetId"]).list_items().items
                )

                if dataset_items and len(dataset_items) > 0:
                    item = dataset_items[0]
                    # Try different possible field names for markdown (in order of likelihood)
                    markdown = (
                        item.get("markdown")
                        or item.get("markdownText")
                        or item.get("textMarkdown")
                        or item.get("markdownContent")
                        or item.get("contentMarkdown")
                        or item.get("text")
                        or ""
                    )
                    if markdown and markdown.strip():
                        logger.info(
                            f"Successfully fetched {len(markdown)} characters of markdown from {url}"
                        )
                        return markdown.strip()
                    else:
                        logger.warning(
                            f"No markdown field found in Apify response for {url}. "
                            f"Available fields: {list(item.keys())}"
                        )
                        return ""
                else:
                    logger.warning(f"No markdown returned from Apify for {url}")
                    return ""

            else:
                # Using cheerio-scraper for simple HTML fetching
                run_input = {
                    "startUrls": [{"url": url}],
                    "maxRequestsPerCrawl": 1,
                    "maxCrawlingDepth": 0,
                    "pageFunction": """async function pageFunction(context) {
                        return {
                            url: context.request.url,
                            html: context.body
                        };
                    }""",
                }

                # Run the cheerio-scraper actor
                run = self.client.actor("apify/cheerio-scraper").call(
                    run_input=run_input
                )

                # Fetch results
                dataset_items = (
                    self.client.dataset(run["defaultDatasetId"]).list_items().items
                )

                if dataset_items and len(dataset_items) > 0:
                    html = dataset_items[0].get("html", "")
                    logger.info(
                        f"Successfully fetched {len(html)} characters from {url}"
                    )
                    return html
                else:
                    logger.warning(f"No content returned from Apify for {url}")
                    return ""

        except Exception as e:
            logger.exception(f"Error fetching page via Apify: {e}")
            return ""

    def extract_sitemap_xml_urls(self, base_url):
        """
        Extracts URLs from sitemap.xml file.

        Tries common sitemap locations:
        - /sitemap.xml
        - /sitemap_index.xml
        - /sitemaps.xml

        Handles both standard sitemap format and sitemap index format.

        Args:
            base_url (str): Base URL of the site

        Returns:
            list: List of unique absolute URLs, or empty list if sitemap not found
        """
        try:
            parsed_base = urlparse(base_url)
            base_domain = parsed_base.netloc
            base_scheme = parsed_base.scheme or "https"

            # Common sitemap locations to try
            sitemap_paths = [
                "/sitemap.xml",
                "/sitemap_index.xml",
                "/sitemaps.xml",
                "/sitemap/sitemap.xml",
            ]

            all_urls = set()

            for sitemap_path in sitemap_paths:
                sitemap_url = f"{base_scheme}://{base_domain}{sitemap_path}"
                logger.info(f"Trying sitemap: {sitemap_url}")

                try:
                    # Fetch sitemap with timeout
                    response = requests.get(
                        sitemap_url, timeout=10, allow_redirects=True
                    )
                    if response.status_code != 200:
                        continue

                    # Parse XML
                    try:
                        root = ET.fromstring(response.content)
                    except ET.ParseError:
                        logger.warning(f"Failed to parse XML from {sitemap_url}")
                        continue

                    # Handle sitemap index format (sitemaps that link to other sitemaps)
                    # Namespace for sitemap XML
                    namespaces = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

                    # Check if it's a sitemap index
                    sitemapindex = root.find("sm:sitemapindex", namespaces)
                    if sitemapindex is None:
                        # Try without namespace
                        sitemapindex = root.find("sitemapindex")

                    if sitemapindex is not None:
                        # It's a sitemap index - extract child sitemap URLs
                        logger.info(f"Found sitemap index at {sitemap_url}")
                        sitemap_locs = sitemapindex.findall(
                            ".//sm:sitemap/sm:loc", namespaces
                        ) or sitemapindex.findall(".//sitemap/loc")

                        for loc in sitemap_locs:
                            child_sitemap_url = loc.text.strip()
                            logger.info(f"Fetching child sitemap: {child_sitemap_url}")
                            try:
                                child_response = requests.get(
                                    child_sitemap_url, timeout=10
                                )
                                if child_response.status_code == 200:
                                    child_root = ET.fromstring(child_response.content)
                                    urls = self._extract_urls_from_sitemap(
                                        child_root, namespaces, base_domain
                                    )
                                    all_urls.update(urls)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to fetch child sitemap {child_sitemap_url}: {e}"
                                )
                    else:
                        # Standard sitemap format
                        logger.info(f"Found standard sitemap at {sitemap_url}")
                        urls = self._extract_urls_from_sitemap(
                            root, namespaces, base_domain
                        )
                        all_urls.update(urls)

                    # If we found URLs, we're done
                    if all_urls:
                        logger.info(
                            f"Successfully extracted {len(all_urls)} URLs from sitemap.xml"
                        )
                        return list(all_urls)

                except requests.RequestException as e:
                    logger.debug(f"Sitemap not found at {sitemap_url}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing sitemap {sitemap_url}: {e}")
                    continue

            logger.info("No sitemap.xml found, will fall back to HTML navigation")
            return []

        except Exception as e:
            logger.exception(f"Error extracting sitemap.xml URLs: {e}")
            return []

    def _extract_urls_from_sitemap(self, root, namespaces, base_domain=None):
        """
        Helper method to extract URLs from a sitemap XML root element.

        Args:
            root: XML root element
            namespaces: XML namespaces dict
            base_domain: Optional domain to filter URLs (only same-domain URLs)

        Returns:
            set: Set of URLs
        """
        urls = set()
        # Try with namespace first
        url_elements = root.findall(".//sm:url/sm:loc", namespaces) or root.findall(
            ".//url/loc"
        )

        for loc in url_elements:
            if loc.text:
                url = loc.text.strip()
                # Remove fragments
                clean_url = url.split("#")[0]
                if clean_url:
                    # Filter to same domain if base_domain is provided
                    if base_domain:
                        parsed = urlparse(clean_url)
                        if parsed.netloc == base_domain:
                            urls.add(clean_url)
                    else:
                        urls.add(clean_url)

        return urls

    def extract_sitemap_urls(self, html, base_url):
        """
        Parses navigation elements to collect unique same-domain URLs.

        This is the FALLBACK method when sitemap.xml is not available.

        Targets: <nav>, <header>, <footer>, <menu>

        Args:
            html (str): Raw HTML content
            base_url (str): Base URL for resolving relative links

        Returns:
            list: List of unique absolute URLs
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            unique_urls = set()
            base_domain = urlparse(base_url).netloc

            # Target semantic navigation elements
            nav_elements = soup.find_all(["nav", "header", "footer", "menu"])

            logger.info(f"Found {len(nav_elements)} navigation elements in {base_url}")

            # If no semantic tags found, fall back to all links
            if not nav_elements:
                logger.warning("No semantic navigation elements found, using all links")
                nav_elements = [soup]

            # Extract links from navigation elements
            for element in nav_elements:
                for a_tag in element.find_all("a", href=True):
                    href = a_tag["href"]

                    # Resolve to absolute URL
                    absolute_url = urljoin(base_url, href)

                    # Check if same domain
                    parsed = urlparse(absolute_url)
                    if parsed.netloc == base_domain:
                        # Remove fragments
                        clean_url = absolute_url.split("#")[0]
                        if clean_url:  # Avoid empty URLs
                            unique_urls.add(clean_url)

            logger.info(
                f"Extracted {len(unique_urls)} unique URLs from HTML navigation"
            )
            return list(unique_urls)

        except Exception as e:
            logger.exception(f"Error extracting sitemap URLs: {e}")
            return []


class HTMLCleaner:
    """Utility for cleaning HTML and extracting visible text."""

    def clean(self, raw_html):
        """
        Cleans HTML by removing scripts, styles, and non-visible elements.
        Keeps semantic tags like <p>, <h1-6>, <li>, <a>.

        Args:
            raw_html (str): Raw HTML content

        Returns:
            tuple: (clean_html, page_text)
        """
        try:
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove unwanted tags
            for tag in soup(
                [
                    "script",
                    "style",
                    "noscript",
                    "iframe",
                    "svg",
                    "meta",
                    "link",
                    "button",
                    "input",
                    "form",
                ]
            ):
                tag.decompose()

            # Extract visible text
            text = soup.get_text(separator=" ", strip=True)

            # Clean up whitespace
            text = " ".join(text.split())

            # Get cleaned HTML (structure preserved)
            clean_html = str(soup)

            return clean_html, text

        except Exception as e:
            logger.exception(f"Error cleaning HTML: {e}")
            return raw_html, ""

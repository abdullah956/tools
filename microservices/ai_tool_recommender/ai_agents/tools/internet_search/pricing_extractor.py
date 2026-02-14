"""Pricing extraction service for AI tools."""

import logging
import re
from typing import Any, Dict, List, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PricingExtractor:
    """Service for extracting pricing information from AI tool websites."""

    def __init__(self):
        """Initialize the pricing extractor."""
        self.pricing_keywords = [
            "pricing",
            "price",
            "cost",
            "plan",
            "subscription",
            "billing",
            "free",
            "premium",
            "pro",
            "enterprise",
            "starter",
            "basic",
            "monthly",
            "yearly",
            "annual",
            "per month",
            "per year",
        ]

        self.pricing_patterns = [
            r"\$[\d,]+(?:\.\d{2})?",  # $99, $99.99, $1,000
            r"€[\d,]+(?:\.\d{2})?",  # €99, €99.99
            r"£[\d,]+(?:\.\d{2})?",  # £99, £99.99
            r"[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP)",  # 99 USD, 99.99 EUR
            r"Free",  # Free
            r"Free Trial",  # Free Trial
            r"Contact Sales",  # Contact Sales
            r"Custom Pricing",  # Custom Pricing
        ]

    async def extract_pricing_from_url(self, url: str) -> Dict[str, Any]:
        """Extract pricing information from a website URL.

        Args:
            url: Website URL to extract pricing from

        Returns:
            Dictionary with pricing information
        """
        try:
            logger.info(f"Extracting pricing from: {url}")

            # Fetch page content
            page_content = await self._fetch_page_content(url)
            if not page_content:
                return {"pricing_found": False, "error": "Failed to fetch page content"}

            # Get page content with links for social media extraction
            content_with_links, links = await self._fetch_page_with_links(url)

            # Extract pricing information
            pricing_info = self._extract_pricing_from_content(page_content, url)

            # Extract social media links
            if content_with_links and links:
                social_links = self._extract_social_media_links(
                    content_with_links, links
                )
                pricing_info["social_media_links"] = social_links

            # If we found pricing buttons/links, try to follow them for more detailed pricing
            if pricing_info.get("pricing_buttons") and not pricing_info.get(
                "price_from"
            ):
                pricing_info = await self._follow_pricing_links(url, pricing_info)

            return pricing_info

        except Exception as e:
            logger.error(f"Error extracting pricing from {url}: {e}")
            return {"pricing_found": False, "error": str(e)}

    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch page content from URL."""
        try:
            timeout_config = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        return soup.get_text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    async def _fetch_page_with_links(self, url: str) -> tuple[Optional[str], List[str]]:
        """Fetch page content and extract all links."""
        try:
            timeout_config = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Extract all links
                        links = []
                        for link in soup.find_all("a", href=True):
                            href = link["href"]
                            # Convert relative URLs to absolute
                            if href.startswith("/"):
                                from urllib.parse import urljoin

                                href = urljoin(url, href)
                            elif not href.startswith(("http://", "https://")):
                                continue
                            links.append(href)

                        return soup.get_text(), links
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None, []

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None, []

    async def _follow_pricing_links(
        self, base_url: str, pricing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Follow pricing links to get detailed pricing information."""
        try:
            logger.info(f"Following pricing links from: {base_url}")

            # Get page content with links
            content, links = await self._fetch_page_with_links(base_url)
            if not content or not links:
                return pricing_info

            # Look for pricing-related links
            pricing_links = []
            for link in links:
                link_lower = link.lower()
                if any(
                    keyword in link_lower
                    for keyword in [
                        "pricing",
                        "price",
                        "plan",
                        "subscription",
                        "billing",
                    ]
                ):
                    pricing_links.append(link)

            # Try to fetch pricing from pricing pages
            for pricing_link in pricing_links[:3]:  # Limit to 3 pricing links
                try:
                    logger.info(f"Following pricing link: {pricing_link}")
                    pricing_content = await self._fetch_page_content(pricing_link)
                    if pricing_content:
                        # Extract pricing from the pricing page
                        pricing_page_info = self._extract_pricing_from_content(
                            pricing_content, pricing_link
                        )
                        if pricing_page_info.get("price_from"):
                            pricing_info["price_from"] = pricing_page_info["price_from"]
                            pricing_info["price_to"] = pricing_page_info["price_to"]
                            pricing_info["pricing_details"].extend(
                                pricing_page_info.get("pricing_details", [])
                            )
                            pricing_info["pricing_source_url"] = pricing_link
                            logger.info(
                                f"Found pricing on {pricing_link}: {pricing_info['price_from']} - {pricing_info['price_to']}"
                            )
                            break
                except Exception as e:
                    logger.warning(f"Error following pricing link {pricing_link}: {e}")
                    continue

            return pricing_info

        except Exception as e:
            logger.error(f"Error following pricing links: {e}")
            return pricing_info

    def _extract_social_media_links(
        self, content: str, links: List[str]
    ) -> Dict[str, str]:
        """Extract social media links from page content and links."""
        social_links = {}

        # Social media patterns
        social_patterns = {
            "twitter": [r"twitter\.com/([^/\s]+)", r"x\.com/([^/\s]+)"],
            "facebook": [r"facebook\.com/([^/\s]+)", r"fb\.com/([^/\s]+)"],
            "linkedin": [r"linkedin\.com/(?:company/|in/)?([^/\s]+)"],
            "instagram": [r"instagram\.com/([^/\s]+)"],
            "youtube": [r"youtube\.com/(?:c/|channel/|user/)?([^/\s]+)"],
            "tiktok": [r"tiktok\.com/@([^/\s]+)"],
        }

        # Check links for social media
        for link in links:
            for platform, patterns in social_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, link, re.IGNORECASE)
                    if match and platform not in social_links:
                        social_links[platform] = link
                        break

        # Also check content for social media mentions
        content_lower = content.lower()
        for platform, patterns in social_patterns.items():
            if platform not in social_links:
                for pattern in patterns:
                    match = re.search(pattern, content_lower)
                    if match:
                        # Try to construct the URL
                        username = match.group(1)
                        if platform == "twitter":
                            social_links[platform] = f"https://twitter.com/{username}"
                        elif platform == "facebook":
                            social_links[platform] = f"https://facebook.com/{username}"
                        elif platform == "linkedin":
                            social_links[
                                platform
                            ] = f"https://linkedin.com/company/{username}"
                        elif platform == "instagram":
                            social_links[platform] = f"https://instagram.com/{username}"
                        elif platform == "youtube":
                            social_links[platform] = f"https://youtube.com/c/{username}"
                        elif platform == "tiktok":
                            social_links[platform] = f"https://tiktok.com/@{username}"
                        break

        return social_links

    def _extract_pricing_from_content(self, content: str, url: str) -> Dict[str, Any]:
        """Extract pricing information from page content.

        Args:
            content: Page content text
            url: Original URL

        Returns:
            Dictionary with pricing information
        """
        try:
            pricing_info = {
                "pricing_found": False,
                "price_from": "",
                "price_to": "",
                "pricing_details": [],
                "pricing_buttons": [],
                "source_url": url,
            }

            # Convert content to lowercase for pattern matching
            # Look for pricing sections
            pricing_sections = self._find_pricing_sections(content)

            # Extract pricing patterns
            pricing_patterns = self._extract_pricing_patterns(content)

            # Look for pricing buttons/links
            pricing_buttons = self._find_pricing_buttons(content)

            if pricing_sections or pricing_patterns or pricing_buttons:
                pricing_info["pricing_found"] = True
                pricing_info["pricing_details"] = pricing_sections
                pricing_info["pricing_buttons"] = pricing_buttons

                # Extract price range
                if pricing_patterns:
                    prices = self._parse_prices(pricing_patterns)
                    if prices:
                        pricing_info["price_from"] = prices[0]
                        if len(prices) > 1:
                            pricing_info["price_to"] = prices[-1]
                        else:
                            pricing_info["price_to"] = prices[0]

                # If no explicit prices found but we have pricing sections/buttons, extract from context
                if not pricing_info.get("price_from") and (
                    pricing_sections or pricing_buttons
                ):
                    extracted_prices = self._extract_prices_from_context(
                        pricing_sections + pricing_buttons
                    )
                    if extracted_prices:
                        pricing_info["price_from"] = extracted_prices[0]
                        pricing_info["price_to"] = (
                            extracted_prices[-1]
                            if len(extracted_prices) > 1
                            else extracted_prices[0]
                        )

            # If no pricing found, check for common pricing indicators
            if not pricing_info["pricing_found"]:
                pricing_info = self._check_common_pricing_indicators(
                    content, pricing_info
                )

            logger.info(
                f"Pricing extraction result for {url}: {pricing_info['pricing_found']}"
            )
            return pricing_info

        except Exception as e:
            logger.error(f"Error extracting pricing from content: {e}")
            return {"pricing_found": False, "error": str(e)}

    def _find_pricing_sections(self, content: str) -> List[str]:
        """Find sections that likely contain pricing information."""
        pricing_sections = []

        # Split content into lines
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Check if line contains pricing keywords
            if any(keyword in line_lower for keyword in self.pricing_keywords):
                # Get context around the line
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                section = "\n".join(lines[start:end]).strip()

                if (
                    len(section) > 10 and len(section) < 500
                ):  # Reasonable section length
                    pricing_sections.append(section)

        return pricing_sections[:5]  # Limit to 5 sections

    def _extract_pricing_patterns(self, content: str) -> List[str]:
        """Extract pricing patterns from content."""
        patterns = []

        for pattern in self.pricing_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            patterns.extend(matches)

        return list(set(patterns))  # Remove duplicates

    def _find_pricing_buttons(self, content: str) -> List[str]:
        """Find pricing-related buttons or links."""
        pricing_buttons = []

        # Common pricing button texts
        button_texts = [
            "get started",
            "start free",
            "free trial",
            "upgrade",
            "subscribe",
            "choose plan",
            "select plan",
            "pricing",
            "buy now",
            "sign up",
            "try free",
            "free plan",
            "premium",
            "pro plan",
            "enterprise",
        ]

        lines = content.split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if (
                any(button_text in line_lower for button_text in button_texts)
                and len(line.strip()) > 5
                and len(line.strip()) < 100
            ):
                pricing_buttons.append(line.strip())

        return pricing_buttons[:5]  # Limit to 5 buttons

    def _parse_prices(self, price_patterns: List[str]) -> List[str]:
        """Parse and clean price patterns."""
        prices = []

        for pattern in price_patterns:
            # Clean up the price
            price = pattern.strip()

            # Skip if it's just text without numbers
            if not re.search(r"\d", price):
                continue

            # Skip very long strings
            if len(price) > 50:
                continue

            # Clean up price format
            price = self._clean_price_format(price)
            if price:
                prices.append(price)

        # Sort prices numerically if possible
        try:

            def price_sort_key(price):
                # Extract numeric value for sorting
                numbers = re.findall(r"[\d,]+(?:\.\d{2})?", price)
                if numbers:
                    return float(numbers[0].replace(",", ""))
                return 0

            prices.sort(key=price_sort_key)
        except Exception:
            pass  # Keep original order if sorting fails

        return prices

    def _clean_price_format(self, price: str) -> str:
        """Clean and standardize price format."""
        # Remove extra spaces
        price = re.sub(r"\s+", "", price)

        # Fix common formatting issues
        # Remove leading zeros from numbers (but keep $0)
        if price.startswith("$0") and len(price) > 2 and price[2].isdigit():
            price = "$" + price[2:]

        # Ensure proper decimal format
        if "." in price and not price.endswith(".00"):
            # Keep as is
            pass
        elif re.match(r"\$\d+$", price):
            # Add .00 for whole numbers
            price = price + ".00"

        return price

    def _extract_prices_from_context(self, context_items: List[str]) -> List[str]:
        """Extract prices from context items like pricing sections and buttons."""
        prices = []

        for item in context_items:
            item_lower = item.lower()

            # Check for explicit "Free" mentions
            if "free" in item_lower and (
                "no cost" in item_lower
                or "no credit card" in item_lower
                or "no watermark" in item_lower
            ):
                prices.append("Free")
                continue

            # Check for trial mentions
            if "trial" in item_lower and "free" in item_lower:
                prices.append("Free Trial")
                continue

            # Check for contact sales
            if any(
                phrase in item_lower
                for phrase in ["contact", "sales", "custom", "quote"]
            ):
                prices.append("Contact Sales")
                continue

            # Look for currency patterns in the context
            currency_patterns = [
                r"\$[\d,]+(?:\.\d{2})?",  # $99, $99.99, $1,000
                r"€[\d,]+(?:\.\d{2})?",  # €99, €99.99
                r"£[\d,]+(?:\.\d{2})?",  # £99, £99.99
                r"[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP)",  # 99 USD, 99.99 EUR
            ]

            for pattern in currency_patterns:
                matches = re.findall(pattern, item, re.IGNORECASE)
                if matches:
                    prices.extend(matches)
                    break

        # Remove duplicates and return
        return list(set(prices))

    def _check_common_pricing_indicators(
        self, content: str, pricing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for common pricing indicators when no explicit pricing is found."""
        content_lower = content.lower()

        # Check for free indicators
        if any(
            indicator in content_lower for indicator in ["free", "no cost", "gratis"]
        ):
            pricing_info["pricing_found"] = True
            pricing_info["price_from"] = "Free"
            pricing_info["price_to"] = "Free"
            pricing_info["pricing_details"] = ["Free service detected"]

        # Check for "Get Started for Free" or similar patterns
        elif any(
            pattern in content_lower
            for pattern in [
                "get started for free",
                "start for free",
                "free ai video editor",
                "no credit card required",
                "no watermark",
            ]
        ):
            pricing_info["pricing_found"] = True
            pricing_info["price_from"] = "Free"
            pricing_info["price_to"] = "Contact for Pricing"
            pricing_info["pricing_details"] = [
                "Free tier available with premium options"
            ]

        # Check for contact sales indicators
        elif any(
            indicator in content_lower
            for indicator in ["contact sales", "custom pricing", "quote"]
        ):
            pricing_info["pricing_found"] = True
            pricing_info["price_from"] = "Contact Sales"
            pricing_info["price_to"] = "Custom Pricing"
            pricing_info["pricing_details"] = ["Custom pricing - contact sales"]

        # Check for trial indicators
        elif any(indicator in content_lower for indicator in ["trial", "demo", "test"]):
            pricing_info["pricing_found"] = True
            pricing_info["price_from"] = "Free Trial"
            pricing_info["price_to"] = "Contact for Pricing"
            pricing_info["pricing_details"] = ["Free trial available"]

        return pricing_info

    async def enhance_tool_with_pricing(
        self, tool_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance tool data with pricing information if missing.

        Args:
            tool_data: Tool data dictionary

        Returns:
            Enhanced tool data with pricing information
        """
        try:
            # Check if pricing is already available
            current_price_from = tool_data.get("Price From", "").strip()
            current_price_to = tool_data.get("Price To", "").strip()

            if current_price_from and current_price_to:
                logger.info(
                    f"Pricing already available for {tool_data.get('Title', 'Unknown')}"
                )
                return tool_data

            # Get website URL
            website_url = tool_data.get("Website", "").strip()
            if not website_url:
                logger.warning(
                    f"No website URL for {tool_data.get('Title', 'Unknown')}"
                )
                return tool_data

            # Extract pricing from website
            pricing_info = await self.extract_pricing_from_url(website_url)

            if pricing_info.get("pricing_found", False):
                # Update tool data with extracted pricing
                tool_data["Price From"] = pricing_info.get("price_from", "")
                tool_data["Price To"] = pricing_info.get("price_to", "")

                # Add pricing source information
                tool_data["Pricing Source"] = "Website Extraction"
                tool_data["Pricing Details"] = pricing_info.get("pricing_details", [])

                logger.info(
                    f"Enhanced {tool_data.get('Title', 'Unknown')} with pricing: {pricing_info.get('price_from')} - {pricing_info.get('price_to')}"
                )
            else:
                logger.info(
                    f"No pricing found for {tool_data.get('Title', 'Unknown')} at {website_url}"
                )

            # Update social media links if found
            social_links = pricing_info.get("social_media_links", {})
            if social_links:
                tool_data["Twitter"] = social_links.get(
                    "twitter", tool_data.get("Twitter", "")
                )
                tool_data["Facebook"] = social_links.get(
                    "facebook", tool_data.get("Facebook", "")
                )
                tool_data["Linkedin"] = social_links.get(
                    "linkedin", tool_data.get("Linkedin", "")
                )
                tool_data["Instagram"] = social_links.get(
                    "instagram", tool_data.get("Instagram", "")
                )
                tool_data["Youtube"] = social_links.get(
                    "youtube", tool_data.get("Youtube", "")
                )
                tool_data["Tiktok"] = social_links.get(
                    "tiktok", tool_data.get("Tiktok", "")
                )

                logger.info(
                    f"Enhanced {tool_data.get('Title', 'Unknown')} with social media links: {list(social_links.keys())}"
                )

            return tool_data

        except Exception as e:
            logger.error(f"Error enhancing tool with pricing: {e}")
            return tool_data

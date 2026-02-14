"""Shared tool data formatting utilities."""

from typing import Any, Dict, List


class ToolDataFormatter:
    """Shared utility for formatting tool data consistently."""

    @classmethod
    def format_tool_data(
        cls,
        tool_data: Dict[str, Any],
        title: str = "",
        url: str = "",
        query: str = "",
        source: str = "Unknown",
    ) -> Dict[str, Any]:
        """Format tool data into standard structure.

        Args:
            tool_data: Raw tool data
            title: Tool title
            url: Tool URL
            query: Search query
            source: Data source

        Returns:
            Formatted tool data
        """
        return {
            "Title": cls._clean_tool_name(
                tool_data.get("tool_name", tool_data.get("Title", title))
            ),
            "Description": tool_data.get(
                "description", tool_data.get("Description", "")
            ),
            "Category": tool_data.get("category", tool_data.get("Category", "AI Tool")),
            "Features": tool_data.get("features", tool_data.get("Features", "")),
            "Tags (Keywords)": tool_data.get("Tags (Keywords)")
            or tool_data.get("tags")
            or (query if query and len(query) < 50 else ""),
            "Website": tool_data.get("website", tool_data.get("Website", url)),
            "Twitter": tool_data.get("social_links", {}).get(
                "twitter", tool_data.get("Twitter", "")
            ),
            "Facebook": tool_data.get("social_links", {}).get(
                "facebook", tool_data.get("Facebook", "")
            ),
            "Linkedin": tool_data.get("social_links", {}).get(
                "linkedin", tool_data.get("Linkedin", "")
            ),
            "Instagram": tool_data.get("social_links", {}).get(
                "instagram", tool_data.get("Instagram", "")
            ),
            "Price From": tool_data.get("pricing", tool_data.get("Price From", "")),
            "Price To": tool_data.get("Price To", ""),
            "Source": source,
            "Relevance Score": tool_data.get(
                "relevance_score", tool_data.get("Relevance Score", 0)
            ),
        }

    @staticmethod
    def format_workflow_tool_data(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Format tool data for workflow nodes.

        Args:
            tool: Tool data dictionary

        Returns:
            Formatted tool data for workflow
        """
        return {
            "label": tool.get("Title", "Unknown Tool"),
            "type": tool.get("Category", "Unknown"),
            "category": [tool.get("Category", "Unknown")],
            "description": tool.get("Description", ""),
            "features": (
                tool.get("Features", "").split(",")
                if isinstance(tool.get("Features"), str)
                else []
            ),
            "tags": (
                tool.get("Tags (Keywords)", "").split(",")
                if isinstance(tool.get("Tags (Keywords)"), str)
                else []
            ),
            "website": tool.get("Website", ""),
            "twitter": tool.get("Twitter", ""),
            "facebook": tool.get("Facebook", ""),
            "linkedin": tool.get("Linkedin", ""),
            "instagram": tool.get("Instagram", ""),
            "price_from": tool.get("Price From", ""),
            "price_to": tool.get("Price To", ""),
            "source": tool.get("Source", "Unknown"),
            "config": {},
        }

    @staticmethod
    def prepare_tools_data_for_prompt(tools: List[Dict[str, Any]]) -> str:
        """Prepare tools data for LLM prompts.

        Args:
            tools: List of tool data

        Returns:
            Formatted string for LLM prompt
        """
        return "\n".join(
            [
                f"Tool {i + 1}: {tool.get('Title', 'Unknown')}\n"
                f"  Category: {tool.get('Category', 'Unknown')}\n"
                f"  Description: {tool.get('Description', '')}\n"
                f"  Features: {tool.get('Features', '')}\n"
                f"  Tags: {tool.get('Tags (Keywords)', '')}\n"
                f"  Website: {tool.get('Website', '')}\n"
                f"  Twitter: {tool.get('Twitter', '')}\n"
                f"  Facebook: {tool.get('Facebook', '')}\n"
                f"  LinkedIn: {tool.get('Linkedin', '')}\n"
                f"  Instagram: {tool.get('Instagram', '')}\n"
                f"  Price From: {tool.get('Price From', '')}\n"
                f"  Price To: {tool.get('Price To', '')}\n"
                f"  Source: {tool.get('Source', 'Unknown')}\n"
                f"  Relevance Score: {tool.get('Relevance Score', tool.get('Similarity Score', 0))}\n"
                for i, tool in enumerate(tools)
            ]
        )

    @staticmethod
    def _clean_tool_name(name: str) -> str:
        """Clean tool name by removing domains and applying proper formatting.

        E.g., 'hubspot.com' -> 'Hubspot'
        """
        if not name:
            return ""

        import re

        # Remove common domains and protocols
        cleaned = name.lower().strip()
        cleaned = re.sub(r"^(https?://)?(www\.)?", "", cleaned)

        # Generic domain stripping: if it contains a dot, take the part before the first dot
        if "." in cleaned:
            cleaned = cleaned.split(".")[0]

        cleaned = cleaned.split("/")[0]  # Remove paths

        # General capitalization: converts "hubspot" to "Hubspot", "fireflies-ai" to "Fireflies Ai"
        return " ".join(
            word.capitalize()
            for word in cleaned.replace("-", " ").replace("_", " ").split()
        )

"""LLM-based tool extraction service for focused internet search."""

import logging

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class ToolRequirementExtractor:
    """Extract specific tool requirements from user queries using LLM."""

    def __init__(self):
        """Initialize the tool requirement extractor."""
        self.llm = get_shared_llm()

    async def extract_tool_requirements(self, query: str) -> dict:
        """
        Extract specific tool requirements from user query.

        Args:
            query: User's search query

        Returns:
            Dictionary with:
            - tool_names: List of specific tool names mentioned
            - tool_types: List of tool categories/types needed
            - requirements: List of specific requirements
            - search_queries: Optimized search queries for finding tools
        """
        prompt = f"""
You are an AI tool discovery expert. Analyze this user query and extract SPECIFIC tool requirements.

User Query: "{query}"

Your task:
1. Identify if the user is looking for SPECIFIC tools (by name) or TYPES of tools (by function)
2. Extract tool names if mentioned
3. Determine what TYPE of tools they need (e.g., "HR automation", "email marketing", "CRM")
4. Generate focused search queries to find OFFICIAL TOOL WEBSITES (not blogs, not lists, not reviews)

Return ONLY valid JSON in this format:
{{
    "has_specific_tools": true/false,
    "tool_names": ["Tool Name 1", "Tool Name 2"],
    "tool_types": ["HR automation software", "employee management platform"],
    "requirements": ["automate HR", "manage employees", "track performance"],
    "search_queries": [
        "HR automation software official site",
        "employee management platform pricing"
    ]
}}

Rules:
- If user mentions specific tool names → set has_specific_tools=true and list them
- If user describes what they want to do → set has_specific_tools=false and describe tool types
- search_queries should target OFFICIAL PRODUCT PAGES, not blogs/reviews/lists
- Add filters like: -blog -article -review -"best" -"top" -"list" to exclude junk
- Maximum 5 search queries
- Be SPECIFIC - "HR software" is better than "software"

Return ONLY the JSON, no explanations.
"""

        try:
            response = await self.llm.generate_response(prompt)

            # Parse JSON response
            import json

            # Clean response
            cleaned = response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()

            # Find JSON content
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_content = cleaned[start_idx : end_idx + 1]
                result = json.loads(json_content)

                logger.info(f"✅ Extracted tool requirements: {result}")
                return result
            else:
                logger.warning("Failed to extract JSON from LLM response")
                return self._get_fallback_extraction(query)

        except Exception as e:
            logger.error(f"Error extracting tool requirements: {e}")
            return self._get_fallback_extraction(query)

    def _get_fallback_extraction(self, query: str) -> dict:
        """Fallback extraction if LLM fails."""
        # Simple keyword-based extraction
        query_lower = query.lower()

        # Common tool types
        tool_types = []
        if "hr" in query_lower or "human resource" in query_lower:
            tool_types.append("HR management software")
        if "crm" in query_lower or "customer" in query_lower:
            tool_types.append("CRM software")
        if "email" in query_lower:
            tool_types.append("email automation tool")
        if "marketing" in query_lower:
            tool_types.append("marketing automation platform")

        # Generic search queries
        search_queries = [
            f'"{query}" software official site -blog -article -review',
            f'"{query}" platform pricing features -blog',
        ]

        return {
            "has_specific_tools": False,
            "tool_names": [],
            "tool_types": tool_types if tool_types else ["automation software"],
            "requirements": [query],
            "search_queries": search_queries,
        }


# Global instance
tool_extractor = ToolRequirementExtractor()

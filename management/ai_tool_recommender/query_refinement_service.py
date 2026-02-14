"""Query refinement service - Simple button-click query refinement."""

import logging
from typing import Dict

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class QueryRefinementService:
    """Service for refining user queries when they click the refine button."""

    def __init__(self):
        """Initialize the query refinement service."""
        self.llm = get_shared_llm()

    async def refine_query(self, user_query: str) -> Dict[str, str]:
        """
        Refine user query to make it more specific and actionable.

        Args:
            user_query: The original user query to refine

        Returns:
            Dict with refined_query, original_query, and explanation
        """
        try:
            refinement_prompt = f"""
            You are a query refinement expert. Your task is to take a user's query and make it more specific,
            actionable, and optimized for finding AI tools and building workflows.

            Original Query: "{user_query}"

            Please refine this query to:
            1. Be more specific and detailed
            2. Include relevant technical terms
            3. Mention key functionality or features needed
            4. Be optimized for AI tool discovery
            5. Keep it concise (1-2 sentences max)

            Return ONLY the refined query, without any explanation or prefix.
            """

            refined_query = await self.llm.generate_response(refinement_prompt)
            refined_query = refined_query.strip().strip('"').strip("'")

            logger.info(f"✅ Refined query: '{user_query}' → '{refined_query}'")

            return {
                "refined_query": refined_query,
                "original_query": user_query,
                "explanation": "Refined to be more specific and actionable",
            }

        except Exception as e:
            logger.error(f"❌ Error refining query: {e}")
            return {
                "refined_query": user_query,
                "original_query": user_query,
                "explanation": "Unable to refine query, using original",
            }

"""Shared LLM service for AI agents."""

import logging

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class SharedLLMService:
    """Shared LLM service to avoid repetition across AI agents."""

    def __init__(self):
        """Initialize the shared LLM service with speed optimizations."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Fastest model
            temperature=0.1,
            max_tokens=4000,  # Increased for complex workflow generation
            request_timeout=120,  # Reduced to 2 minutes to prevent gateway timeouts
            max_retries=2,  # Balanced retries for reliability
            streaming=False,  # Disable streaming for faster completion
        )
        logger.info("Shared LLM service initialized with speed optimizations")

    async def generate_response(self, prompt: str) -> str:
        """Generate response from LLM.

        Args:
            prompt: The prompt to send to LLM

        Returns:
            LLM response text
        """
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise

    async def parse_json_response(self, response_text: str) -> dict:
        """Parse JSON response from LLM, handling markdown code blocks.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON data
        """
        import json

        # Clean and parse JSON response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing error: {json_error}")
            raise


# Global instance
shared_llm = SharedLLMService()

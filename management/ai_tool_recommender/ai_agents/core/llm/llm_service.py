"""Shared LLM service for AI agents."""

import logging

from django.conf import settings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class SharedLLMService:
    """Shared LLM service to avoid repetition across AI agents."""

    def __init__(self):
        """Initialize the shared LLM service with Claude 4.5 Sonnet."""
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5",
            anthropic_api_key=settings.CLAUDE_API_KEY,
            temperature=0.1,
            max_tokens=12000,
            timeout=600,  # Claude uses 'timeout' instead of 'request_timeout'
            max_retries=2,
        )
        logger.info("Shared LLM service initialized with claude-sonnet-4-5")

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
        except RuntimeError as e:
            # Handle executor shutdown errors (common during Django reloads)
            if (
                "cannot schedule new futures after shutdown" in str(e)
                or "shutdown" in str(e).lower()
            ):
                logger.warning(
                    "LLM executor shutdown detected, recreating LLM instance"
                )
                # Recreate the LLM instance with fresh executor
                self.llm = ChatAnthropic(
                    model="claude-sonnet-4-5",
                    anthropic_api_key=settings.CLAUDE_API_KEY,
                    temperature=0.1,
                    max_tokens=12000,
                    timeout=600,
                    max_retries=2,
                )
                # Retry once with fresh instance
                try:
                    response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                    return response.content.strip()
                except Exception as retry_error:
                    logger.error(
                        f"Retry with fresh LLM instance also failed: {retry_error}"
                    )
                    raise RuntimeError(
                        "LLM service temporarily unavailable. Please try again in a moment."
                    ) from retry_error
            else:
                raise
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
            logger.error(f"Failed JSON text (truncated): {response_text[:300]}...")

            # Additional cleanup for common LLM JSON errors (like unescaped quotes)
            try:
                # Basic usage of regex to fix common "unterminated string" issues if they are trailing commas or similar
                # For now, let's try a second pass with strict=False equivalent if possible, or just fail cleanly
                logger.info(
                    "Attempting to parse with strict=False or cleaning controls"
                )
                return json.loads(response_text, strict=False)
            except Exception:
                pass

            raise


# Global instance
shared_llm = SharedLLMService()


def get_shared_llm():
    """Get the shared LLM service instance."""
    return shared_llm

"""Base agent class for all conversational agents."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all conversational agents."""

    def __init__(self):
        """Initialize the base agent."""
        self.agent_name = self.get_agent_name()
        logger.info(f"✅ Initialized {self.agent_name}")

    @abstractmethod
    def get_agent_name(self) -> str:
        """
        Get the name of this agent.

        Returns:
            Agent name string
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        user_message: str,
        conversation,
        workflow_id: str,
        request_user,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process a user message and return a response.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response dictionary with message, tools, workflow, etc.
        """
        pass

    @abstractmethod
    async def can_handle(
        self, user_message: str, conversation, current_state: str
    ) -> bool:
        """
        Check if this agent can handle the current message/state.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state

        Returns:
            True if agent can handle, False otherwise
        """
        pass

    def format_response(
        self,
        message: str,
        tools: list = None,
        workflow_changes: dict = None,
        suggestions: list = None,
        **extra_fields,
    ) -> Dict[str, Any]:
        """
        Format agent response in standard structure.

        Args:
            message: AI response message
            tools: List of tools mentioned
            workflow_changes: Changes made to workflow
            suggestions: Suggested actions
            **extra_fields: Additional fields to include

        Returns:
            Formatted response dictionary
        """
        response = {
            "message": message,
            "tools_mentioned": tools or [],
            "workflow_changes": workflow_changes or {},
            "suggestions": suggestions or [],
            "agent": self.agent_name,
        }

        # Add any extra fields
        response.update(extra_fields)

        return response

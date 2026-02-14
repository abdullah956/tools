"""Agent Router - Routes messages to appropriate agents."""

import logging
from typing import Any, Dict, Optional

from ai_tool_recommender.agents.general_assistant import GeneralAssistant
from ai_tool_recommender.agents.implementation_chat import ImplementationChat
from ai_tool_recommender.agents.refine_query_agent import RefineQueryAgent
from ai_tool_recommender.agents.tool_assistant import ToolAssistant
from ai_tool_recommender.agents.workflow_builder_agent import WorkflowBuilderAgent

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Routes user messages to the appropriate agent.

    Selection is based on:
    1. Explicit agent parameter in request.
    2. Current conversation state.
    3. Message content analysis.

    Available Agents:
    - refine_query_generator: Asks 10 questions to build refined query
    - workflow_builder: Builds workflows from refined queries
    - general_assistant: Provides general guidance and help
    - implementation_chat: Provides implementation guidance
    - tool_assistant: Answers questions about tools
    """

    def __init__(self):
        """Initialize the agent router with all available agents."""
        self.agents = {
            "refine_query_generator": RefineQueryAgent(),
            "workflow_builder": WorkflowBuilderAgent(),
            "general_assistant": GeneralAssistant(),
            "implementation_chat": ImplementationChat(),
            "tool_assistant": ToolAssistant(),
        }
        logger.info(f"✅ AgentRouter initialized with {len(self.agents)} agents")

    async def route_message(
        self,
        user_message: str,
        conversation,
        workflow_id: str,
        request_user,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Route user message to the appropriate agent.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            agent_name: Optional explicit agent name from request
            **kwargs: Additional parameters to pass to agent

        Returns:
            Response dictionary from the selected agent
        """
        try:
            # Step 1: Check if agent is explicitly specified
            if agent_name:
                selected_agent = self._get_agent_by_name(agent_name)
                if selected_agent:
                    logger.info(f"🎯 Explicitly routing to: {selected_agent.agent_name}")
                    return await selected_agent.process_message(
                        user_message, conversation, workflow_id, request_user, **kwargs
                    )
                else:
                    logger.warning(
                        f"⚠️ Unknown agent '{agent_name}', falling back to auto-routing"
                    )

            # Step 2: Auto-route based on conversation state and message
            selected_agent = await self._auto_select_agent(
                user_message, conversation, conversation.workflow_state, **kwargs
            )

            if not selected_agent:
                # Fallback to refine_query_generator for initial state
                logger.warning(
                    "⚠️ No agent selected, defaulting to refine_query_generator"
                )
                selected_agent = self.agents["refine_query_generator"]

            logger.info(f"🎯 Auto-routing to: {selected_agent.agent_name}")

            # Process message with selected agent
            response = await selected_agent.process_message(
                user_message, conversation, workflow_id, request_user, **kwargs
            )

            # Ensure agent name is in response
            if "agent" not in response:
                response["agent"] = selected_agent.agent_name

            return response

        except Exception as e:
            logger.error(f"❌ Error in AgentRouter.route_message: {e}", exc_info=True)
            # Return error response
            return {
                "message": "I encountered an error. Please try again.",
                "tools_mentioned": [],
                "workflow_changes": {},
                "suggestions": ["Try again", "Start over"],
                "agent": "error",
            }

    async def _auto_select_agent(
        self, user_message: str, conversation, current_state: str, **kwargs
    ):
        """
        Automatically select agent based on state and message.

        Priority (in order):
        1. Check each agent's can_handle() method
        2. Analyze message content for keywords
        3. Use state-based routing
        4. Default to general_assistant

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state
            **kwargs: Additional parameters (may include tool_id)

        Returns:
            Selected agent or None
        """
        try:
            # Priority 1: Let each agent decide if it can handle the message
            # Check in priority order: specific agents first, general last

            # Check implementation_chat (very specific)
            impl_agent = self.agents["implementation_chat"]
            if await impl_agent.can_handle(
                user_message, conversation, current_state, **kwargs
            ):
                logger.info("🎯 implementation_chat can handle this")
                return impl_agent

            # Check tool_assistant (specific)
            tool_agent = self.agents["tool_assistant"]
            if await tool_agent.can_handle(
                user_message, conversation, current_state, **kwargs
            ):
                logger.info("🎯 tool_assistant can handle this")
                return tool_agent

            # Check refine_query_generator (state-dependent)
            refine_agent = self.agents["refine_query_generator"]
            if await refine_agent.can_handle(
                user_message, conversation, current_state, **kwargs
            ):
                logger.info("🎯 refine_query_generator can handle this")
                return refine_agent

            # Check workflow_builder (state-dependent)
            workflow_agent = self.agents["workflow_builder"]
            if await workflow_agent.can_handle(
                user_message, conversation, current_state, **kwargs
            ):
                logger.info("🎯 workflow_builder can handle this")
                return workflow_agent

            # Check general_assistant (catches general queries)
            general_agent = self.agents["general_assistant"]
            if await general_agent.can_handle(
                user_message, conversation, current_state, **kwargs
            ):
                logger.info("🎯 general_assistant can handle this")
                return general_agent

            # Priority 2: Keyword-based routing (fallback)
            message_lower = user_message.lower()

            # Implementation keywords
            if any(
                keyword in message_lower
                for keyword in [
                    "implement",
                    "setup",
                    "configure",
                    "integrate",
                    "deploy",
                ]
            ):
                return impl_agent

            # Tool keywords
            if (
                any(
                    keyword in message_lower
                    for keyword in ["what is", "tell me about", "explain", "compare"]
                )
                and "tool" in message_lower
            ):
                return tool_agent

            # Refine query keywords
            if any(
                keyword in message_lower
                for keyword in ["ask me questions", "questionnaire", "refine"]
            ):
                return refine_agent

            # Workflow builder keywords
            if any(
                keyword in message_lower
                for keyword in [
                    "generate workflow",
                    "create workflow",
                    "build workflow",
                ]
            ):
                return workflow_agent

            # Priority 3: State-based routing
            if current_state in ["initial", "questioning"]:
                return refine_agent
            elif current_state == "workflow_ready":
                # In workflow_ready, prefer general_assistant for ambiguous messages
                return general_agent

            # Priority 4: Default to general_assistant
            logger.info("🎯 Defaulting to general_assistant")
            return general_agent

        except Exception as e:
            logger.error(f"Error in auto-select: {e}", exc_info=True)
            return None

    def _get_agent_by_name(self, agent_name: str):
        """
        Get agent by name.

        Args:
            agent_name: Name of the agent

        Returns:
            Agent instance or None
        """
        return self.agents.get(agent_name)

    def get_available_agents(self) -> list:
        """
        Get list of available agent names.

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def get_agent_info(self) -> Dict[str, str]:
        """
        Get information about all agents.

        Returns:
            Dictionary mapping agent names to descriptions
        """
        return {
            "refine_query_generator": "Asks 10 questions to build a detailed refined query",
            "workflow_builder": "Builds workflows from refined queries or single statements",
            "general_assistant": "Provides general guidance, help, and navigation",
            "implementation_chat": "Provides step-by-step implementation guidance",
            "tool_assistant": "Answers questions about tools and their role in workflows",
        }

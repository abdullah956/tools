"""General Assistant Agent - Handles simple conversational chat with tool search capability."""

import logging
from typing import Any, Dict

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.tools.ai_tool_recommender import AIToolRecommender
from ai_tool_recommender.conversational_service import ConversationAI

logger = logging.getLogger(__name__)


class GeneralAssistant(BaseAgent):
    """
    Agent that handles simple conversational chat with tool search capability.

    This agent:
    - Provides simple conversational responses
    - Can search for tools using Pinecone and Internet
    - Can provide information about specific tools
    - Works whether workflow exists or not
    - Conversational about tools and workflows
    - Only returns message field, nothing else
    """

    def __init__(self):
        """Initialize the general assistant agent."""
        super().__init__()
        self.conversation_ai = ConversationAI()
        self.recommender = AIToolRecommender()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "general_assistant"

    async def can_handle(
        self, user_message: str, conversation, current_state: str, **kwargs
    ) -> bool:
        """
        Check if this agent should handle the message.

        This agent handles general conversation, but avoids conflicts with:
        - tool_assistant (specific tool questions)
        - implementation_chat (implementation questions)
        - workflow_builder (add/remove commands)
        - refine_query_generator (questionnaire)

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state
            **kwargs: Additional parameters (max_results, include_pinecone, etc.)

        Returns:
            True if agent can handle
        """
        message_lower = user_message.lower()

        # Avoid conflicts with other agents
        # Don't handle if it's a specific action command
        action_keywords = ["add", "remove", "delete", "include", "take out"]
        if any(keyword in message_lower for keyword in action_keywords):
            return False

        # Don't handle if it's specifically about implementation
        if "implement" in message_lower or "how to build" in message_lower:
            return False

        # Don't handle if it's specifically asking about a tool in workflow
        # (tool_assistant handles this)
        if "tool" in message_lower and conversation.workflow_nodes:
            # Check if asking about specific tool in workflow
            workflow_labels = [
                node.get("data", {}).get("label", "").lower()
                for node in conversation.workflow_nodes
            ]
            if any(label in message_lower for label in workflow_labels if label):
                return False

        # Handle general conversation (including tool searches)
        return True

    async def process_message(
        self,
        user_message: str,
        conversation,
        workflow_id: str,
        request_user,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process user message with simple conversational chat and tool search.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response dictionary with only message field
        """
        try:
            current_state = conversation.workflow_state
            has_workflow = bool(conversation.workflow_nodes)
            message_lower = user_message.lower()

            logger.info(
                f"🤖 {self.agent_name} processing message in state: {current_state}, has_workflow: {has_workflow}"
            )

            # Check if user is asking about tools
            tool_query_keywords = [
                "what are the tools",
                "tools for",
                "find tools",
                "search for tools",
                "recommend tools",
                "suggest tools",
                "what tools",
                "which tools",
                "tools that",
                "tell me about",
                "information about",
            ]
            is_tool_query = any(
                keyword in message_lower for keyword in tool_query_keywords
            ) or (
                "tool" in message_lower
                and any(
                    word in message_lower for word in ["for", "about", "what", "which"]
                )
            )

            tools = []  # Initialize tools list
            if is_tool_query:
                # Search for tools
                logger.info(f"🔍 User is asking about tools, performing search...")
                search_result = await self.recommender.search_tools(
                    query=user_message,
                    max_results=5,
                    include_pinecone=True,
                    include_internet=True,
                )

                if search_result.get("status") == "success":
                    tools = search_result.get("tools", [])[:5]  # Limit to 5 tools

                if tools:
                    # Format tools conversationally
                    tool_descriptions = []
                    for tool in tools:
                        name = tool.get("Title", tool.get("title", "Unknown"))
                        desc = tool.get("Description", tool.get("description", ""))
                        website = tool.get("Website", tool.get("website", ""))
                        if desc:
                            tool_descriptions.append(
                                f"**{name}**: {desc[:150]}{'...' if len(desc) > 150 else ''}"
                            )
                        elif website:
                            tool_descriptions.append(f"**{name}**: {website}")
                        else:
                            tool_descriptions.append(f"**{name}**")

                    response_message = (
                        f"I found {len(tools)} tool(s) that might help:\n\n"
                        + "\n\n".join(tool_descriptions)
                    )
                else:
                    response_message = (
                        f"I searched for tools related to '{user_message}', but couldn't find any specific results. "
                        "You might want to try rephrasing your query or be more specific about what you're looking for."
                    )
            else:
                # Regular conversational response
                context = {
                    "state": current_state,
                    "has_workflow": has_workflow,
                    "workflow_nodes": conversation.workflow_nodes
                    if has_workflow
                    else [],
                    "original_query": conversation.original_query,
                    "chat_history": conversation.chat_history[-5:]
                    if conversation.chat_history
                    else [],
                }

                # Generate conversational response using LLM
                response_message = (
                    await self.conversation_ai.generate_conversational_response(
                        message=user_message,
                        intent="general_chat",
                        context=context,
                    )
                )

            # Return only message field, no suggestions or other fields
            return self.format_response(
                message=response_message,
                suggestions=[],  # Empty suggestions
                tools_mentioned=[
                    tool.get("Title", tool.get("title", "")) for tool in tools
                ]
                if tools
                else [],
            )

        except Exception as e:
            logger.error(
                f"❌ Error in {self.agent_name}.process_message: {e}", exc_info=True
            )
            return self.format_response(
                message="I'm here to help! What would you like to chat about?",
                suggestions=[],  # Empty suggestions
            )

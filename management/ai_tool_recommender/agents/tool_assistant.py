"""Tool Assistant Agent - Answers questions about tools and their role in workflows."""

import logging
from typing import Any, Dict, List, Optional

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.tools.ai_tool_recommender import AIToolRecommender

logger = logging.getLogger(__name__)


class ToolAssistant(BaseAgent):
    """
    Agent that helps users understand tools and their role in workflows.

    This agent:
    - Answers questions about specific tools
    - Explains tool features and capabilities
    - Describes how tools fit in the workflow
    - Compares tools
    - Provides tool recommendations
    - Explains tool integrations
    """

    def __init__(self):
        """Initialize the tool assistant agent."""
        super().__init__()
        self.llm = get_shared_llm()
        self.recommender = AIToolRecommender()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "tool_assistant"

    async def can_handle(
        self, user_message: str, conversation, current_state: str, **kwargs
    ) -> bool:
        """
        Check if this agent should handle the message.

        This agent handles:
        - Questions about specific tools
        - Tool comparison requests
        - Tool feature inquiries
        - Tool role in workflow questions
        - When tool_id is provided in kwargs (always handles)

        IMPORTANT: Only works AFTER workflow is generated (requires workflow_nodes).

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state
            **kwargs: Additional parameters (may include tool_id)

        Returns:
            True if agent can handle
        """
        # If tool_id is provided, always handle (focused conversation)
        if kwargs.get("tool_id"):
            if conversation.workflow_nodes:
                return True
            return False

        # CRITICAL: Only handle if workflow exists
        if not conversation.workflow_nodes:
            return False

        message_lower = user_message.lower()

        # Tool-related keywords
        tool_keywords = [
            "what is",
            "tell me about",
            "explain",
            "how does",
            "what does",
            "why",
            "compare",
            "difference between",
            "vs",
            "versus",
            "alternative to",
            "similar to",
            "features of",
            "capabilities of",
            "use case",
            "when to use",
        ]

        # Check if asking about tools
        has_tool_keyword = any(keyword in message_lower for keyword in tool_keywords)
        mentions_tool = "tool" in message_lower

        # Check if message mentions any tool from workflow
        tool_names = [
            node.get("data", {}).get("label", "").lower()
            for node in conversation.workflow_nodes
        ]
        mentions_workflow_tool = any(
            tool_name in message_lower for tool_name in tool_names if tool_name
        )

        if has_tool_keyword and (mentions_tool or mentions_workflow_tool):
            return True

        # General tool questions (only if workflow exists)
        if has_tool_keyword and mentions_tool:
            return True

        return False

    async def process_message(
        self,
        user_message: str,
        conversation,
        workflow_id: str,
        request_user,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process user message about tools.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters (may include tool_id)

        Returns:
            Response dictionary
        """
        try:
            tool_id = kwargs.get("tool_id")
            logger.info(
                f"🤖 {self.agent_name} processing tool question"
                + (f" (focused on tool_id: {tool_id})" if tool_id else "")
            )

            # Check if workflow exists (required for tool_assistant)
            if not conversation.workflow_nodes:
                return self.format_response(
                    message="You need to generate a workflow first before I can help you with tools. Once you have a workflow, I can explain each tool, compare them, and answer questions about their role.",
                    suggestions=[],
                )

            # If tool_id is provided, focus conversation on that specific tool
            if tool_id:
                tool_node = await self._find_tool_by_id(tool_id, conversation)
                if not tool_node:
                    return self.format_response(
                        message=f"I couldn't find a tool with ID '{tool_id}' in your workflow. Please check the tool ID and try again.",
                        suggestions=[],
                        tool_id=tool_id,  # Include tool_id even in error response
                    )

                # All questions are about this specific tool
                response = await self._handle_tool_focused_conversation(
                    user_message, tool_node, conversation, workflow_id
                )
                # Ensure tool_id is included in response
                if "tool_id" not in response:
                    response["tool_id"] = tool_id
                return response

            # Determine the type of tool question (when no tool_id provided)
            question_type = await self._analyze_tool_question_type(
                user_message, conversation
            )

            if question_type == "specific_tool":
                return await self._handle_specific_tool_question(
                    user_message, conversation, workflow_id
                )
            elif question_type == "tool_comparison":
                return await self._handle_tool_comparison(
                    user_message, conversation, workflow_id
                )
            elif question_type == "tool_role":
                return await self._handle_tool_role_question(
                    user_message, conversation, workflow_id
                )
            elif question_type == "tool_features":
                return await self._handle_tool_features_question(
                    user_message, conversation, workflow_id
                )
            elif question_type == "tool_search":
                return await self._handle_tool_search(
                    user_message, conversation, workflow_id
                )
            else:
                return await self._handle_general_tool_question(
                    user_message, conversation, workflow_id
                )

        except Exception as e:
            logger.error(
                f"❌ Error in {self.agent_name}.process_message: {e}", exc_info=True
            )
            return self.format_response(
                message="I can help you understand tools in your workflow. What would you like to know?",
                suggestions=[],
            )

    async def _analyze_tool_question_type(self, user_message: str, conversation) -> str:
        """
        Analyze what type of tool question the user is asking.

        Args:
            user_message: User's question
            conversation: ConversationSession object

        Returns:
            Question type string
        """
        try:
            message_lower = user_message.lower()

            # Check for comparison
            if any(
                keyword in message_lower
                for keyword in [
                    "compare",
                    "vs",
                    "versus",
                    "difference between",
                    "alternative",
                ]
            ):
                return "tool_comparison"

            # Check for role/purpose questions
            if any(
                keyword in message_lower
                for keyword in ["why", "role", "purpose", "why use", "why do i need"]
            ):
                return "tool_role"

            # Check for features
            if any(
                keyword in message_lower
                for keyword in ["features", "capabilities", "what can", "what does"]
            ):
                return "tool_features"

            # Check if asking about specific tool in workflow
            if conversation.workflow_nodes:
                tool_names = [
                    node.get("data", {}).get("label", "").lower()
                    for node in conversation.workflow_nodes
                ]
                if any(
                    tool_name in message_lower for tool_name in tool_names if tool_name
                ):
                    return "specific_tool"

            # Check for tool search
            if any(
                keyword in message_lower
                for keyword in ["find", "search", "recommend", "suggest", "what tools"]
            ):
                return "tool_search"

            return "general_tool"

        except Exception as e:
            logger.error(f"Error analyzing tool question type: {e}", exc_info=True)
            return "general_tool"

    async def _handle_specific_tool_question(
        self, user_message: str, conversation, workflow_id: str
    ) -> Dict[str, Any]:
        """Handle questions about a specific tool."""
        try:
            # Find which tool user is asking about
            tool_node = await self._identify_tool_from_message(
                user_message, conversation
            )

            if not tool_node:
                return self.format_response(
                    message="I couldn't identify which tool you're asking about. Could you be more specific?",
                    suggestions=[],
                )

            tool_data = tool_node.get("data", {})
            tool_name = tool_data.get("label", "Unknown Tool")

            # Generate detailed response about the tool
            response_message = await self._generate_tool_explanation(
                user_message=user_message,
                tool_data=tool_data,
                workflow_nodes=conversation.workflow_nodes,
                workflow_edges=conversation.workflow_edges,
                original_query=conversation.original_query,
            )

            return self.format_response(
                message=response_message,
                suggestions=[],
                tool_mentioned=tool_name,
            )

        except Exception as e:
            logger.error(f"Error handling specific tool question: {e}", exc_info=True)
            return self.format_response(
                message="I can explain any tool in your workflow. Which one would you like to know about?",
                suggestions=[],
            )

    async def _handle_tool_comparison(
        self, user_message: str, conversation, workflow_id: str
    ) -> Dict[str, Any]:
        """Handle tool comparison requests."""
        try:
            # Extract tools to compare
            tools_to_compare = await self._extract_tools_for_comparison(
                user_message, conversation
            )

            if len(tools_to_compare) < 2:
                return self.format_response(
                    message="I need at least two tools to compare. Which tools would you like me to compare?",
                    suggestions=[],
                )

            # Generate comparison
            comparison_response = await self._generate_tool_comparison(
                tools_to_compare, user_message, conversation
            )

            return self.format_response(
                message=comparison_response,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error handling tool comparison: {e}", exc_info=True)
            return self.format_response(
                message="I can compare tools for you. Which tools would you like to compare?",
                suggestions=[],
            )

    async def _handle_tool_role_question(
        self, user_message: str, conversation, workflow_id: str
    ) -> Dict[str, Any]:
        """Handle questions about tool's role in workflow."""
        try:
            # Find which tool user is asking about
            tool_node = await self._identify_tool_from_message(
                user_message, conversation
            )

            if not tool_node:
                return self.format_response(
                    message="Which tool's role would you like me to explain?",
                    suggestions=[],
                )

            tool_data = tool_node.get("data", {})

            # Generate role explanation
            role_explanation = await self._generate_tool_role_explanation(
                tool_data=tool_data,
                workflow_nodes=conversation.workflow_nodes,
                workflow_edges=conversation.workflow_edges,
                original_query=conversation.original_query,
            )

            return self.format_response(
                message=role_explanation,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error handling tool role question: {e}", exc_info=True)
            return self.format_response(
                message="I can explain why each tool is in your workflow. Which tool would you like to know about?",
                suggestions=[],
            )

    async def _handle_tool_features_question(
        self,
        user_message: str,
        conversation,
        workflow_id: str,
        tool_node: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Handle questions about tool features."""
        try:
            # Use provided tool_node if available, otherwise find it
            if not tool_node:
                tool_node = await self._identify_tool_from_message(
                    user_message, conversation
                )

            if not tool_node:
                return self.format_response(
                    message="Which tool's features would you like to know about?",
                    suggestions=[],
                )

            tool_data = tool_node.get("data", {})
            tool_name = tool_data.get("label", "Unknown Tool")
            features_raw = tool_data.get("features", [])
            description = tool_data.get("description", "")
            website = tool_data.get("website", "")

            # Handle features: can be string (Internet Search) or list (Pinecone)
            if isinstance(features_raw, str):
                # Convert comma-separated string to list
                features = [f.strip() for f in features_raw.split(",") if f.strip()]
            elif isinstance(features_raw, list):
                features = features_raw
            else:
                features = []

            # Check if this is an Internet Search tool with redirect URL
            is_redirect_url = (
                "vertexaisearch.cloud.google.com" in website
                or "grounding-api-redirect" in website
            )

            # Generate features explanation using LLM for better formatting
            prompt = f"""
            Generate a helpful explanation about this tool's features and capabilities.

            **Tool Name:** {tool_name}
            **Description:** {description}
            **Features:** {', '.join(features) if features else 'Not specified'}
            **Source:** {tool_data.get('source', 'Unknown')}

            Create a clear, informative response that:
            1. Explains what the tool does
            2. Lists key features in a readable format
            3. Highlights main capabilities
            4. Is conversational and helpful

            Do NOT include the redirect URL in your response.
            Format with Markdown (bold, bullets, etc.).

            Return ONLY the explanation.
            """

            try:
                features_text = await self.llm.generate_response(prompt)
            except Exception as e:
                logger.error(f"Error generating features explanation: {e}")
                # Fallback to simple format
                features_text = f"**{tool_name}**\n\n{description}\n\n"
                if features:
                    features_text += "**Key Features:**\n"
                    for i, feature in enumerate(features[:10], 1):
                        features_text += f"- {feature}\n"
                else:
                    features_text += "Feature details are being gathered."

            # Only add website link if it's NOT a redirect URL
            if website and not is_redirect_url:
                features_text += f"\n\n**Learn More:** {website}"

            return self.format_response(
                message=features_text,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error handling tool features question: {e}", exc_info=True)
            return self.format_response(
                message="I can tell you about tool features. Which tool would you like to know about?",
                suggestions=[],
            )

    async def _handle_tool_search(
        self, user_message: str, conversation, workflow_id: str
    ) -> Dict[str, Any]:
        """Handle tool search requests."""
        try:
            # Search for tools
            search_result = await self.recommender.search_tools(
                query=user_message,
                max_results=5,
                include_pinecone=True,
                include_internet=True,
            )

            tools = search_result.get("tools", [])

            if not tools:
                return self.format_response(
                    message="I couldn't find tools matching your request. Could you be more specific?",
                    suggestions=[],
                )

            # Format tool list
            tools_text = f"I found {len(tools)} tools:\n\n"
            for i, tool in enumerate(tools, 1):
                tool_name = tool.get("Title") or tool.get("title") or "Unknown"
                tool_desc = tool.get("Description") or tool.get("description") or ""
                tools_text += f"**{i}. {tool_name}**\n{tool_desc[:200]}...\n\n"

            tools_text += "Would you like to know more about any of these?"

            return self.format_response(
                message=tools_text,
                suggestions=[],
                tools_mentioned=[
                    tool.get("Title") or tool.get("title") for tool in tools
                ],
            )

        except Exception as e:
            logger.error(f"Error handling tool search: {e}", exc_info=True)
            return self.format_response(
                message="I can search for tools. What are you looking for?",
                suggestions=[],
            )

    async def _handle_general_tool_question(
        self, user_message: str, conversation, workflow_id: str
    ) -> Dict[str, Any]:
        """Handle general tool questions."""
        try:
            # Use LLM to generate response
            workflow_tools = [
                node.get("data", {}).get("label", "Unknown")
                for node in conversation.workflow_nodes
            ]

            prompt = f"""
            User is asking a general question about tools.

            **User's Question:** "{user_message}"

            **Tools in Workflow:** {', '.join(workflow_tools) if workflow_tools else 'No workflow yet'}

            Generate a helpful response that:
            1. Answers their question
            2. Provides relevant information about tools
            3. Suggests next steps
            4. Is conversational and friendly

            Return ONLY the response message.
            """

            response = await self.llm.generate_response(prompt)

            return self.format_response(
                message=response.strip(),
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error handling general tool question: {e}", exc_info=True)
            return self.format_response(
                message="I can help you understand tools. What would you like to know?",
                suggestions=[],
            )

    async def _identify_tool_from_message(
        self, user_message: str, conversation
    ) -> Optional[Dict]:
        """Identify which tool the user is asking about."""
        try:
            message_lower = user_message.lower()

            # Check each tool in workflow
            for node in conversation.workflow_nodes:
                tool_name = node.get("data", {}).get("label", "").lower()
                if tool_name and tool_name in message_lower:
                    return node

            return None

        except Exception as e:
            logger.error(f"Error identifying tool: {e}", exc_info=True)
            return None

    async def _extract_tools_for_comparison(
        self, user_message: str, conversation
    ) -> List[Dict]:
        """Extract tools to compare from message."""
        try:
            tools = []
            message_lower = user_message.lower()

            # Check workflow tools
            for node in conversation.workflow_nodes:
                tool_name = node.get("data", {}).get("label", "").lower()
                if tool_name and tool_name in message_lower:
                    tools.append(node.get("data", {}))

            return tools

        except Exception as e:
            logger.error(f"Error extracting tools for comparison: {e}", exc_info=True)
            return []

    async def _generate_tool_explanation(
        self,
        user_message: str,
        tool_data: Dict,
        workflow_nodes: List,
        workflow_edges: List,
        original_query: str,
    ) -> str:
        """Generate detailed explanation about a tool."""
        try:
            tool_name = tool_data.get("label", "Unknown Tool")
            description = tool_data.get("description", "")
            features_raw = tool_data.get("features", [])

            # Handle features: can be string (Internet Search) or list (Pinecone)
            if isinstance(features_raw, str):
                features = [f.strip() for f in features_raw.split(",") if f.strip()]
            elif isinstance(features_raw, list):
                features = features_raw
            else:
                features = []

            prompt = f"""
            User asked: "{user_message}"

            Explain this tool in detail:

            **Tool:** {tool_name}
            **Description:** {description}
            **Features:** {', '.join(features[:10]) if features else 'N/A'}
            **Source:** {tool_data.get('source', 'Database')}

            **Workflow Context:**
            - Original Goal: {original_query}
            - Total Tools: {len(workflow_nodes)}

            Generate a comprehensive explanation that:
            1. Describes what the tool does
            2. Explains its key features
            3. Shows how it fits in the workflow
            4. Mentions its benefits
            5. Provides practical use cases

            Be conversational and helpful. Use Markdown formatting.
            Do NOT include any redirect URLs in your response.

            Return ONLY the explanation.
            """

            response = await self.llm.generate_response(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating tool explanation: {e}", exc_info=True)
            return f"I can tell you about {tool_data.get('label', 'this tool')}. What specifically would you like to know?"

    async def _generate_tool_comparison(
        self, tools: List[Dict], user_message: str, conversation
    ) -> str:
        """Generate comparison between tools."""
        try:
            prompt = f"""
            User asked: "{user_message}"

            Compare these tools:

            {self._format_json_for_prompt(tools)}

            Generate a detailed comparison that:
            1. Compares key features
            2. Highlights differences
            3. Mentions strengths and weaknesses
            4. Provides use case recommendations
            5. Suggests which might be better for specific scenarios

            Be objective and helpful. Use Markdown formatting with tables if appropriate.

            Return ONLY the comparison.
            """

            response = await self.llm.generate_response(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating tool comparison: {e}", exc_info=True)
            return "I can compare these tools for you. What specific aspects would you like me to compare?"

    async def _generate_tool_role_explanation(
        self,
        tool_data: Dict,
        workflow_nodes: List,
        workflow_edges: List,
        original_query: str,
    ) -> str:
        """Generate explanation of tool's role in workflow."""
        try:
            tool_name = tool_data.get("label", "Unknown Tool")
            description = tool_data.get("description", "")
            features_raw = tool_data.get("features", [])

            # Handle features: can be string (Internet Search) or list (Pinecone)
            if isinstance(features_raw, str):
                features = [f.strip() for f in features_raw.split(",") if f.strip()]
            elif isinstance(features_raw, list):
                features = features_raw
            else:
                features = []

            prompt = f"""
            Explain the role of {tool_name} in this workflow.

            **Tool:** {tool_name}
            **Description:** {description}
            **Key Features:** {', '.join(features[:5]) if features else 'N/A'}

            **Workflow Context:**
            - Original Goal: {original_query}
            - Total Tools: {len(workflow_nodes)}
            - Connections: {len(workflow_edges)}

            Generate an explanation that:
            1. Explains WHY this tool is in the workflow
            2. Describes its specific role
            3. Shows how it contributes to the goal
            4. Mentions what would be missing without it
            5. Explains how it connects with other tools

            Be clear and concise. Use Markdown formatting.
            Do NOT include any redirect URLs in your response.

            Return ONLY the explanation.
            """

            response = await self.llm.generate_response(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Error generating tool role explanation: {e}", exc_info=True)
            return f"{tool_name} plays an important role in your workflow. It helps accomplish your goals by providing key functionality."

    async def _find_tool_by_id(self, tool_id: str, conversation) -> Optional[Dict]:
        """
        Find a tool node by its ID.

        Args:
            tool_id: Tool node ID (UUID or format like "tool_001")
            conversation: ConversationSession object

        Returns:
            Tool node dictionary or None if not found
        """
        try:
            # Search through workflow nodes
            for node in conversation.workflow_nodes:
                # Check direct ID match
                if node.get("id") == tool_id:
                    return node

                # Check if tool_id matches original_id in data
                node_data = node.get("data", {})
                if node_data.get("original_id") == tool_id:
                    return node

            return None

        except Exception as e:
            logger.error(f"Error finding tool by ID: {e}", exc_info=True)
            return None

    async def _handle_tool_focused_conversation(
        self,
        user_message: str,
        tool_node: Dict,
        conversation,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Handle conversation focused on a specific tool (when tool_id is provided).

        Args:
            user_message: User's message
            tool_node: The specific tool node to focus on
            conversation: ConversationSession object
            workflow_id: Workflow UUID

        Returns:
            Response dictionary
        """
        try:
            tool_data = tool_node.get("data", {})
            tool_name = tool_data.get("label", "Unknown Tool")

            # Analyze what the user is asking about this specific tool
            message_lower = user_message.lower()

            # Check question type
            if any(keyword in message_lower for keyword in ["why", "role", "purpose"]):
                # Role/purpose question
                response = await self._generate_tool_role_explanation(
                    tool_data=tool_data,
                    workflow_nodes=conversation.workflow_nodes,
                    workflow_edges=conversation.workflow_edges,
                    original_query=conversation.original_query,
                )
            elif any(
                keyword in message_lower
                for keyword in ["features", "capabilities", "what can", "what does"]
            ):
                # Features question
                response = await self._handle_tool_features_question(
                    user_message, conversation, workflow_id, tool_node=tool_node
                )
                return response
            else:
                # General question about the tool
                response = await self._generate_tool_explanation(
                    user_message=user_message,
                    tool_data=tool_data,
                    workflow_nodes=conversation.workflow_nodes,
                    workflow_edges=conversation.workflow_edges,
                    original_query=conversation.original_query,
                )

            return self.format_response(
                message=response,
                suggestions=[],
                tool_mentioned=tool_name,
                tool_id=tool_node.get("id"),  # Include tool_id in response
            )

        except Exception as e:
            logger.error(
                f"Error handling tool-focused conversation: {e}", exc_info=True
            )
            tool_name = tool_node.get("data", {}).get("label", "this tool")
            return self.format_response(
                message=f"I can help you understand {tool_name}. What would you like to know?",
                suggestions=[],
            )

    def _format_json_for_prompt(self, data) -> str:
        """Format JSON data for LLM prompt."""
        try:
            import json

            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

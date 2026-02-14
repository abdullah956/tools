"""Implementation Chat Agent - Provides step-by-step implementation guidance."""

import logging
from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.models import WorkflowImplementationGuide

logger = logging.getLogger(__name__)


class ImplementationChat(BaseAgent):
    """
    Agent that provides implementation guidance for workflows.

    This agent:
    - Takes context of what implementation part user is asking about
    - Uses complete implementation guide
    - Provides step-by-step guidance
    - Answers specific implementation questions
    - Maintains conversation context about implementation
    """

    def __init__(self):
        """Initialize the implementation chat agent."""
        super().__init__()
        self.llm = get_shared_llm()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "implementation_chat"

    async def can_handle(
        self, user_message: str, conversation, current_state: str, **kwargs
    ) -> bool:
        """
        Check if this agent should handle the message.

        This agent handles:
        - Implementation questions
        - "How to implement" queries
        - Step-by-step guidance requests
        - Integration questions
        - When context is provided in payload (always handles)

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state
            **kwargs: Additional parameters (may include context)

        Returns:
            True if agent can handle
        """
        # If context is provided (even if empty string), always handle (focused conversation)
        if "context" in kwargs:
            if conversation.workflow_nodes:
                return True
            return False

        message_lower = user_message.lower()

        # Implementation keywords
        implementation_keywords = [
            "implement",
            "implementation",
            "how to set up",
            "how to configure",
            "how to integrate",
            "step by step",
            "guide me through",
            "walk me through",
            "how do i use",
            "how do i connect",
            "setup",
            "configure",
            "integrate",
            "deploy",
            "install",
        ]

        # Check for implementation keywords
        if (
            any(keyword in message_lower for keyword in implementation_keywords)
            and conversation.workflow_nodes
        ):
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
        Process user message for implementation guidance.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters (may include context)

        Returns:
            Response dictionary
        """
        try:
            # Extract context from payload (can be empty string)
            provided_context = kwargs.get("context", "")
            # Treat empty string as valid context (user explicitly sent context field)
            if "context" in kwargs:
                provided_context = (
                    kwargs.get("context") or ""
                )  # Ensure it's a string even if None
            logger.info(
                f"🤖 {self.agent_name} processing implementation question"
                + (
                    f" (with context: {provided_context[:50]}...)"
                    if provided_context
                    else " (no context provided)"
                )
            )

            # Check if workflow exists
            if not conversation.workflow_nodes:
                return self.format_response(
                    message="You need to generate a workflow first before I can help with implementation.",
                    suggestions=[],
                )

            # Get or generate implementation guide
            implementation_guide = await self._get_implementation_guide(
                workflow_id, conversation, request_user
            )

            if not implementation_guide:
                return self.format_response(
                    message="I'm generating your implementation guide. This may take a moment...",
                    suggestions=[],
                )

            # Analyze what part of implementation user is asking about
            implementation_context = await self._analyze_implementation_context(
                user_message, conversation, implementation_guide
            )

            # Generate contextual response with provided context and ask next questions
            response_message = await self._generate_implementation_response(
                user_message=user_message,
                implementation_guide=implementation_guide,
                implementation_context=implementation_context,
                conversation_history=conversation.chat_history,
                workflow_nodes=conversation.workflow_nodes,
                provided_context=provided_context,
            )

            return self.format_response(
                message=response_message,
                suggestions=[],
                implementation_context=implementation_context,
            )

        except Exception as e:
            logger.error(
                f"❌ Error in {self.agent_name}.process_message: {e}", exc_info=True
            )
            return self.format_response(
                message="I can help you implement your workflow. What specific part would you like guidance on?",
                suggestions=[],
            )

    async def _get_implementation_guide(
        self, workflow_id: str, conversation, request_user
    ) -> Optional[str]:
        """
        Get or generate implementation guide for workflow.

        Args:
            workflow_id: Workflow UUID
            conversation: ConversationSession object
            request_user: User object

        Returns:
            Implementation guide text or None
        """
        try:
            # Try to get existing implementation guide
            guide_obj = await sync_to_async(
                lambda: WorkflowImplementationGuide.objects.filter(
                    workflow_id=workflow_id, user=request_user
                ).first()
            )()

            if guide_obj and guide_obj.status == "completed":
                return guide_obj.implementation_guide

            # If no guide exists or it's not complete, generate one
            if not guide_obj or guide_obj.status != "completed":
                logger.info(
                    f"Generating implementation guide for workflow {workflow_id}"
                )
                return await self._generate_implementation_guide(
                    workflow_id, conversation, request_user
                )

            return None

        except Exception as e:
            logger.error(f"Error getting implementation guide: {e}", exc_info=True)
            return None

    async def _generate_implementation_guide(
        self, workflow_id: str, conversation, request_user
    ) -> Optional[str]:
        """
        Generate a new implementation guide.

        Args:
            workflow_id: Workflow UUID
            conversation: ConversationSession object
            request_user: User object

        Returns:
            Generated implementation guide text
        """
        try:
            workflow_nodes = conversation.workflow_nodes
            workflow_edges = conversation.workflow_edges
            original_query = conversation.original_query

            if not workflow_nodes:
                return None

            # Build tool information
            tools_info = []
            for i, node in enumerate(workflow_nodes, 1):
                node_data = node.get("data", {})
                tools_info.append(
                    {
                        "number": i,
                        "name": node_data.get("label", "Unknown Tool"),
                        "description": node_data.get("description", ""),
                        "features": node_data.get("features", []),
                        "website": node_data.get("website", ""),
                    }
                )

            # Generate implementation guide using LLM
            prompt = f"""
            Generate a comprehensive step-by-step implementation guide for this workflow.

            **Original Goal:** {original_query}

            **Workflow Tools ({len(tools_info)} total):**
            {self._format_json_for_prompt(tools_info)}

            **Tool Connections:**
            {self._format_json_for_prompt(workflow_edges)}

            Generate a detailed implementation guide in Markdown format with the following structure:

            # Implementation Guide

            ## Overview
            [Brief overview of what this workflow accomplishes]

            ## Prerequisites
            - [List of prerequisites]

            ## Step-by-Step Implementation

            ### Step 1: [Tool Name] Setup
            1. [Detailed setup instructions]
            2. [Configuration steps]
            3. [Testing verification]

            ### Step 2: [Next Tool] Integration
            1. [Integration instructions]
            2. [Connection setup]
            3. [Testing verification]

            [Continue for all tools...]

            ## Integration Points
            [Explain how tools connect and data flows]

            ## Testing & Validation
            [How to test the complete workflow]

            ## Troubleshooting
            [Common issues and solutions]

            ## Best Practices
            [Recommendations for optimal use]

            Make it:
            - Detailed and actionable
            - Easy to follow for non-technical users
            - Include specific configuration examples
            - Mention potential gotchas
            - Provide testing steps

            Return ONLY the Markdown formatted guide.
            """

            implementation_guide = await self.llm.generate_response(prompt)

            # Save to database
            await sync_to_async(WorkflowImplementationGuide.objects.update_or_create)(
                workflow_id=workflow_id,
                user=request_user,
                defaults={
                    "implementation_guide": implementation_guide,
                    "workflow_snapshot": {
                        "nodes": workflow_nodes,
                        "edges": workflow_edges,
                    },
                    "tools_count": len(workflow_nodes),
                    "status": "completed",
                },
            )

            logger.info(
                f"✅ Generated and saved implementation guide for workflow {workflow_id}"
            )

            return implementation_guide

        except Exception as e:
            logger.error(f"Error generating implementation guide: {e}", exc_info=True)
            return None

    async def _analyze_implementation_context(
        self, user_message: str, conversation, implementation_guide: str
    ) -> Dict[str, Any]:
        """
        Analyze what part of implementation the user is asking about.

        Args:
            user_message: User's question
            conversation: ConversationSession object
            implementation_guide: Full implementation guide

        Returns:
            Context dictionary with relevant section and tools
        """
        try:
            prompt = f"""
            User is asking about implementation of their workflow.

            **User's Question:** "{user_message}"

            **Implementation Guide (excerpt):**
            {implementation_guide[:2000]}...

            **Workflow Tools:**
            {self._format_json_for_prompt([node.get("data", {}).get("label") for node in conversation.workflow_nodes])}

            Analyze what the user is asking about and return JSON:
            {{
              "section": "setup|integration|testing|troubleshooting|general",
              "specific_tool": "tool name if asking about specific tool, otherwise null",
              "step_number": "step number if asking about specific step, otherwise null",
              "topic": "brief description of what they're asking about"
            }}

            Return ONLY valid JSON.
            """

            response = await self.llm.generate_response(prompt)
            context = await self.llm.parse_json_response(response)

            return context or {
                "section": "general",
                "specific_tool": None,
                "step_number": None,
                "topic": "general implementation",
            }

        except Exception as e:
            logger.error(f"Error analyzing implementation context: {e}", exc_info=True)
            return {
                "section": "general",
                "specific_tool": None,
                "step_number": None,
                "topic": "general implementation",
            }

    async def _generate_implementation_response(
        self,
        user_message: str,
        implementation_guide: str,
        implementation_context: Dict[str, Any],
        conversation_history: list,
        workflow_nodes: list,
        provided_context: str = "",
    ) -> str:
        """
        Generate contextual implementation response with follow-up questions.

        Args:
            user_message: User's question
            implementation_guide: Full implementation guide
            implementation_context: Context about what user is asking
            conversation_history: Chat history
            workflow_nodes: Workflow nodes
            provided_context: Additional context from payload

        Returns:
            Response message with next questions
        """
        try:
            # Get recent conversation for context
            recent_history = conversation_history[-3:] if conversation_history else []
            history_text = "\n".join(
                [
                    f"User: {msg.get('user', '')}\nAI: {msg.get('ai', '')}"
                    for msg in recent_history
                ]
            )

            # Build context section
            context_section = ""
            if provided_context:
                context_section = f"""
            **Additional Context Provided:**
            {provided_context}
            """

            prompt = f"""
            You are an implementation guide assistant helping a user implement their workflow.

            **User's Question:** "{user_message}"
            {context_section}
            **Analyzed Context:**
            - Section: {implementation_context.get('section')}
            - Specific Tool: {implementation_context.get('specific_tool', 'N/A')}
            - Topic: {implementation_context.get('topic')}

            **Recent Conversation:**
            {history_text if history_text else 'No previous conversation'}

            **Full Implementation Guide:**
            {implementation_guide}

            **Workflow Tools:**
            {self._format_json_for_prompt([node.get("data", {}).get("label") for node in workflow_nodes])}

            Generate a helpful, detailed response that:
            1. Processes and incorporates the provided context (if any) along with the user's query
            2. Directly answers their question based on both the context and query
            3. References specific steps from the implementation guide
            4. Provides actionable guidance
            5. Includes examples if relevant
            6. Mentions potential issues to watch for
            7. Is conversational and easy to understand

            **IMPORTANT: After answering, ask 2-3 relevant follow-up questions** that:
            - Help the user proceed with the next steps
            - Clarify any ambiguities in the implementation
            - Guide them through the workflow
            - Are specific and actionable

            Format the response as:
            [Your detailed answer]

            **Next Steps - Questions to consider:**
            1. [First follow-up question]
            2. [Second follow-up question]
            3. [Third follow-up question (if relevant)]

            If they're asking about a specific step, quote that step and explain it.
            If they're asking about a specific tool, focus on that tool's setup and integration.
            If it's a general question, provide an overview with key points.

            Keep the response focused and practical. Use Markdown formatting for clarity.

            Return ONLY the response message with the follow-up questions.
            """

            response = await self.llm.generate_response(prompt)
            return response.strip()

        except Exception as e:
            logger.error(
                f"Error generating implementation response: {e}", exc_info=True
            )
            return (
                "I can help you with implementation. Could you be more specific about "
                "what step or tool you need help with?"
            )

    async def _generate_implementation_suggestions(
        self, implementation_context: Dict[str, Any], conversation
    ) -> list:
        """Generate contextual suggestions for implementation."""
        section = implementation_context.get("section", "general")
        specific_tool = implementation_context.get("specific_tool")

        if section == "setup":
            return [
                "What's the next step?",
                "How do I test this?",
                "Show me integration steps",
            ]
        elif section == "integration":
            return [
                "How do I connect these tools?",
                "What's the data flow?",
                "Show me testing steps",
            ]
        elif section == "testing":
            return [
                "What should I test?",
                "How do I verify it works?",
                "Common issues to check?",
            ]
        elif section == "troubleshooting":
            return [
                "What are common issues?",
                "How do I debug this?",
                "Where can I get help?",
            ]
        elif specific_tool:
            return [
                f"How do I configure {specific_tool}?",
                f"How does {specific_tool} connect?",
                "What's the next tool?",
            ]
        else:
            return [
                "Show me the full guide",
                "How do I start?",
                "What's the first step?",
            ]

    def _format_json_for_prompt(self, data) -> str:
        """Format JSON data for LLM prompt."""
        try:
            import json

            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

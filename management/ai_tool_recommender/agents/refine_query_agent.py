"""Refine Query Agent - Handles questionnaire-based workflow refinement."""

import logging
import time
import uuid
from typing import Any, Dict

from asgiref.sync import sync_to_async

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.tools.ai_tool_recommender import AIToolRecommender
from ai_tool_recommender.conversational_service import ConversationAI
from ai_tool_recommender.models import RefinedQuery
from ai_tool_recommender.questionnaire_service import QuestionnaireService
from ai_tool_recommender.workflow_generation_service import WorkflowGenerationService

logger = logging.getLogger(__name__)


class RefineQueryAgent(BaseAgent):
    """
    Agent that handles refine query generation through a 4-phase questionnaire.

    This agent:
    - Asks 10 questions across 4 phases
    - Builds a comprehensive refined query
    - Handles iterative refinement
    - Saves refined query to database
    """

    def __init__(self):
        """Initialize the refine query agent."""
        super().__init__()
        self.questionnaire_service = QuestionnaireService()
        self.recommender = AIToolRecommender()
        self.conversation_ai = ConversationAI()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "refine_query_generator"

    async def can_handle(
        self, user_message: str, conversation, current_state: str, **kwargs
    ) -> bool:
        """
        Check if this agent should handle the message.

        This agent handles:
        - questioning state (answering questions)
        - initial state ONLY if user wants to create a workflow (not greetings)

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state
            **kwargs: Additional parameters (max_results, include_pinecone, etc.)

        Returns:
            True if agent can handle
        """
        # Always handle questioning state (user is answering questions)
        if current_state == "questioning":
            return True

        # For initial state, only handle if user wants to create a workflow
        # Don't start questionnaire for simple greetings
        if current_state == "initial":
            message_lower = user_message.lower().strip()

            # Skip simple greetings and casual messages
            greetings = [
                "hello",
                "hi",
                "hey",
                "greetings",
                "good morning",
                "good afternoon",
                "good evening",
                "howdy",
                "what's up",
                "sup",
                "yo",
            ]

            if message_lower in greetings or message_lower in [
                g + "!" for g in greetings
            ]:
                return False

            # Only handle if message indicates workflow creation intent
            workflow_keywords = [
                "workflow",
                "automate",
                "tools",
                "need",
                "want",
                "create",
                "build",
                "generate",
                "help me",
                "i want",
                "i need",
                "looking for",
                "find",
                "search",
                "recommend",
                "suggest",
                "questionnaire",
                "ask me questions",
            ]

            # Check if message has workflow-related intent
            has_workflow_intent = any(
                keyword in message_lower for keyword in workflow_keywords
            )

            # Also check if message is substantial (not just a greeting)
            is_substantial = len(message_lower.split()) >= 3

            return has_workflow_intent or is_substantial

        # If workflow_ready but questionnaire is not complete, continue handling
        if current_state == "workflow_ready":
            questionnaire = conversation.questionnaire_json
            if (
                questionnaire
                and not self.questionnaire_service.is_questionnaire_complete(
                    questionnaire
                )
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
        Process user message through questionnaire flow.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response dictionary
        """
        try:
            current_state = conversation.workflow_state

            logger.info(
                f"🤖 {self.agent_name} processing message in state: {current_state}"
            )

            # STATE: initial - Check if this is a greeting, if so redirect to general_assistant
            if current_state == "initial":
                message_lower = user_message.lower().strip()

                # Check if this is a simple greeting
                greetings = [
                    "hello",
                    "hi",
                    "hey",
                    "greetings",
                    "good morning",
                    "good afternoon",
                    "good evening",
                    "howdy",
                    "what's up",
                    "sup",
                    "yo",
                ]

                if message_lower in greetings or message_lower in [
                    g + "!" for g in greetings
                ]:
                    # Redirect to general_assistant for greetings
                    from ai_tool_recommender.agents.general_assistant import (
                        GeneralAssistant,
                    )

                    general_agent = GeneralAssistant()
                    return await general_agent.process_message(
                        user_message, conversation, workflow_id, request_user, **kwargs
                    )

                # Check if message has workflow intent, if not redirect to general_assistant
                workflow_keywords = [
                    "workflow",
                    "automate",
                    "tools",
                    "need",
                    "want",
                    "create",
                    "build",
                    "generate",
                    "help me",
                    "i want",
                    "i need",
                    "looking for",
                    "find",
                    "search",
                    "recommend",
                    "suggest",
                    "questionnaire",
                    "ask me questions",
                ]

                has_workflow_intent = any(
                    keyword in message_lower for keyword in workflow_keywords
                )
                is_substantial = len(message_lower.split()) >= 3

                if not has_workflow_intent and not is_substantial:
                    # Not a workflow request, redirect to general_assistant
                    from ai_tool_recommender.agents.general_assistant import (
                        GeneralAssistant,
                    )

                    general_agent = GeneralAssistant()
                    return await general_agent.process_message(
                        user_message, conversation, workflow_id, request_user, **kwargs
                    )

                # Has workflow intent, start questionnaire
                return await self._handle_initial_state(
                    user_message, conversation, workflow_id, request_user
                )

            # STATE: questioning - Process answer
            elif current_state == "questioning":
                return await self._handle_questioning_state(
                    user_message, conversation, workflow_id, request_user, **kwargs
                )

            # STATE: workflow_ready - Handle all workflow conversations
            elif current_state == "workflow_ready":
                return await self._handle_workflow_ready_state(
                    user_message, conversation, workflow_id, request_user, **kwargs
                )

            else:
                # Fallback
                return self.format_response(
                    message="I'm the refine query generator. Let me help you build a detailed query through some questions.",
                    suggestions=["Start questionnaire", "Ask me anything"],
                )

        except Exception as e:
            logger.error(
                f"❌ Error in {self.agent_name}.process_message: {e}", exc_info=True
            )
            return self.format_response(
                message="I encountered an error. Let's start over with your workflow request.",
                suggestions=["Tell me what you want to build"],
            )

    async def _handle_initial_state(
        self, user_message: str, conversation, workflow_id: str, request_user
    ) -> Dict[str, Any]:
        """
        Handle initial state - start questionnaire.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object

        Returns:
            Response with first question
        """
        try:
            logger.info("📝 Starting questionnaire for refine query generation")

            # Initialize questionnaire
            questionnaire = await self.questionnaire_service.initialize_questionnaire(
                user_message
            )

            # Save to conversation
            conversation.questionnaire_json = questionnaire
            conversation.workflow_state = "questioning"
            await sync_to_async(conversation.save)()

            # Get first question
            first_question = self.questionnaire_service.get_next_question(questionnaire)

            if not first_question:
                raise ValueError("Failed to generate first question")

            # Format response
            message = self._format_question_message(first_question, questionnaire)

            return self.format_response(
                message=message,
                suggestions=first_question.get("examples", []),
                tool_examples=first_question.get("tool_examples", []),
                progress_percentage=questionnaire.get("progress_percentage", 0),
                questionnaire_progress=self.questionnaire_service.get_progress_message(
                    questionnaire
                )[0],
            )

        except Exception as e:
            logger.error(f"❌ Error starting questionnaire: {e}", exc_info=True)
            raise

    async def _handle_questioning_state(
        self, user_message: str, conversation, workflow_id: str, request_user, **kwargs
    ) -> Dict[str, Any]:
        """
        Handle questioning state - process answer and ask next question.

        Args:
            user_message: User's answer
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response with next question or completion
        """
        try:
            questionnaire = conversation.questionnaire_json

            # Process the answer
            (
                updated_questionnaire,
                phase_complete,
                all_complete,
            ) = await self.questionnaire_service.process_answer(
                questionnaire, user_message
            )

            # Update conversation
            conversation.questionnaire_json = updated_questionnaire
            await sync_to_async(conversation.save)()

            # Check if all phases complete
            if all_complete:
                return await self._handle_questionnaire_complete(
                    conversation, workflow_id, request_user, **kwargs
                )

            # Check if phase complete - transition to next phase
            if phase_complete:
                current_phase = updated_questionnaire.get("current_phase", 1)
                logger.info(f"🎉 Phase {current_phase} complete! Transitioning...")

                updated_questionnaire = (
                    await self.questionnaire_service.transition_to_next_phase(
                        updated_questionnaire
                    )
                )

                # Save updated questionnaire
                conversation.questionnaire_json = updated_questionnaire
                await sync_to_async(conversation.save)()

            # Get next question
            next_question = self.questionnaire_service.get_next_question(
                updated_questionnaire
            )

            if not next_question:
                # No more questions - complete
                return await self._handle_questionnaire_complete(
                    conversation, workflow_id, request_user, **kwargs
                )

            # Format next question
            message = self._format_question_message(
                next_question, updated_questionnaire
            )

            return self.format_response(
                message=message,
                suggestions=next_question.get("examples", []),
                tool_examples=next_question.get("tool_examples", []),
                progress_percentage=updated_questionnaire.get("progress_percentage", 0),
                questionnaire_progress=self.questionnaire_service.get_progress_message(
                    updated_questionnaire
                )[0],
                is_refined=next_question.get("is_refined", False),
            )

        except Exception as e:
            logger.error(f"❌ Error processing answer: {e}", exc_info=True)
            raise

    async def _handle_questionnaire_complete(
        self, conversation, workflow_id: str, request_user, **kwargs
    ) -> Dict[str, Any]:
        """
        Handle questionnaire completion - save refined query and auto-generate workflow.

        Args:
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters (max_results, include_pinecone, etc.)

        Returns:
            Completion response with generated workflow
        """
        try:
            logger.info("🎉 Questionnaire complete! Saving refined query...")

            questionnaire = conversation.questionnaire_json
            refined_query = questionnaire.get("workflow_info", {}).get("refined_query")

            if not refined_query:
                raise ValueError("No refined query generated")

            # Save refined query to database
            await self.questionnaire_service.save_refined_query_to_db(
                refined_query=refined_query,
                workflow_id=workflow_id,
                conversation_session=conversation,
                request_user=request_user,
                questionnaire_json=questionnaire,
            )

            logger.info("✅ Refined query saved! Auto-generating workflow...")

            # Automatically generate workflow from refined query
            workflow_response = await self._auto_generate_workflow(
                refined_query=refined_query,
                conversation=conversation,
                workflow_id=workflow_id,
                request_user=request_user,
                **kwargs,
            )

            # Update conversation state to workflow_ready
            conversation.workflow_state = "workflow_ready"
            await sync_to_async(conversation.save)()

            # Merge workflow response with completion message
            message = (
                "Perfect! I've created a detailed refined query and automatically generated your workflow. "
                f"{workflow_response.get('message', '')}"
            )

            return self.format_response(
                message=message,
                tools_mentioned=workflow_response.get("tools_mentioned", []),
                workflow_changes=workflow_response.get("workflow_changes", {}),
                suggestions=workflow_response.get(
                    "suggestions",
                    [
                        "Modify workflow",
                        "Add more tools",
                        "Get implementation guide",
                    ],
                ),
                is_refined=True,
                progress_percentage=100,
            )

        except Exception as e:
            logger.error(f"❌ Error completing questionnaire: {e}", exc_info=True)
            raise

    def _normalize_tool_name(self, name: str) -> str:
        """Normalize tool name for matching.

        Handles:
        - Case insensitivity
        - ".ai" suffix variations (mindhyve.ai -> mindhyve)
        - Common separators
        - Special characters removal
        """
        if not name:
            return ""
        normalized = (
            name.lower()
            .strip()
            .replace(" - ", " ")
            .replace(" | ", " ")
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
            .replace("|", "")
            .replace(":", "")
            .replace("/", "")
        )
        # Remove .ai suffix for matching (mindhyve.ai -> mindhyve)
        # But keep it if it's part of the actual name (e.g., "ai.com" should stay)
        if normalized.endswith("ai") and len(normalized) > 2:
            # Only remove if it's clearly a suffix (not part of domain like "ai.com")
            base = normalized[:-2]
            if base and len(base) >= 3:  # Ensure we have a meaningful base
                normalized = base
        return normalized

    def _create_node_identifier(self, tool_name: str, website: str) -> str:
        """
        Create a unique identifier for workflow nodes to prevent duplicates.
        Uses same logic as WorkflowGenerationService for consistency.

        Args:
            tool_name: Name of the tool
            website: Website URL of the tool

        Returns:
            Unique identifier string
        """
        try:
            # Strategy 1: Use website domain if available (most reliable)
            if website and website.startswith(("http://", "https://")):
                try:
                    from urllib.parse import urlparse

                    domain = urlparse(website).netloc.lower()
                    # Remove www. prefix for consistency
                    if domain.startswith("www."):
                        domain = domain[4:]
                    if domain:
                        return f"domain:{domain}"
                except Exception:
                    pass

            # Strategy 2: Normalize tool name (case-insensitive, remove special chars)
            if tool_name:
                normalized_name = self._normalize_tool_name(tool_name)
                # Only use if we have a meaningful name (at least 3 chars)
                if len(normalized_name) >= 3:
                    return f"name:{normalized_name}"

            # Strategy 3: Fallback to original name (normalized)
            return f"original:{tool_name.lower().strip()}" if tool_name else "unknown"

        except Exception as e:
            logger.error(f"Error creating node identifier: {e}")
            return f"fallback:{tool_name.lower().strip()}" if tool_name else "error"

    async def _extract_tool_names_from_query(self, query: str) -> list:
        """
        Use LLM to extract specific tool names from user query.

        Args:
            query: User's search query

        Returns:
            List of extracted tool names
        """
        try:
            # Quick check: If query is a single word (after cleaning), treat it as a tool name
            cleaned_query = query.strip().lower()
            # Remove common stop words and workflow-related terms
            stop_words = ["add", "include", "to", "my", "workflow", "the", "a", "an"]
            words = [w for w in cleaned_query.split() if w not in stop_words]

            # If only one word remains, it's likely a tool name
            if len(words) == 1:
                tool_name = words[0].strip()
                # Capitalize first letter (common for tool names)
                tool_name = tool_name.capitalize()
                logger.info(
                    f"✅ Single word detected, treating as tool name: {tool_name}"
                )
                return [tool_name]

            # For multi-word queries, use LLM extraction
            from ai_tool_recommender.ai_agents.core.llm.llm_service import (
                get_shared_llm,
            )

            prompt = f"""Extract ALL SPECIFIC tool/software names mentioned in this query.
IMPORTANT: Extract EVERY tool name mentioned, not just some of them.

Query: "{query}"

Rules:
- Return ALL specific tool names mentioned (e.g., "HubSpot", "Zoom", "Fireflies", "DocuSign", "Slack", "Salesforce")
- Do NOT return generic terms (e.g., "payroll tool", "CRM software", "workflow management")
- Look for capitalized words that are likely tool names
- Look for words ending in common tool suffixes: -ly, -ify, -hub, -sign, -desk, -force, -zoom, etc.
- Single-word queries are likely tool names (e.g., "edvenity" -> ["Edvenity"])
- If no specific tools mentioned, return empty list
- Return as JSON array: ["Tool1", "Tool2", "Tool3", ...]

Examples:
Query: "I want to use Gusto and BambooHR for payroll"
Response: ["Gusto", "BambooHR"]

Query: "I have 10 sales reps using HubSpot, Zoom, Fireflies, and DocuSign"
Response: ["HubSpot", "Zoom", "Fireflies", "DocuSign"]

Query: "edvenity"
Response: ["Edvenity"]

Query: "add edvenity"
Response: ["Edvenity"]

Query: "I need a payroll automation tool"
Response: []

Query: "Compare Slack vs Microsoft Teams"
Response: ["Slack", "Microsoft Teams"]

Query: "automate lead generation using Salesforce, Calendly for scheduling, and Mailchimp for emails"
Response: ["Salesforce", "Calendly", "Mailchimp"]

Query: "add workflow management software"
Response: []

CRITICAL: Extract ALL tool names from the query, not just the first one or two. If 5 tools are mentioned, return all 5.

Now extract from the query above. Return ONLY the JSON array, no explanations."""

            llm = get_shared_llm()
            response = await llm.generate_response(prompt)

            # Parse JSON response
            import json
            import re

            # Extract JSON array from response
            json_match = re.search(r"\[.*?\]", response, re.DOTALL)
            if json_match:
                tools = json.loads(json_match.group())
                logger.info(f"✅ LLM extracted tools: {tools}")
                return tools if isinstance(tools, list) else []

            logger.info("ℹ️ No specific tools extracted from query")
            return []

        except Exception as e:
            logger.error(f"Error extracting tool names: {e}")
            return []

    async def _extract_tools_from_refined_query(self, refined_query: str) -> list:
        """Extract tool names mentioned in refined query.

        Args:
            refined_query: Refined query text (may contain tool names)

        Returns:
            List of extracted tool names
        """
        try:
            if not refined_query:
                return []

            from ai_tool_recommender.ai_agents.core.llm.llm_service import (
                get_shared_llm,
            )

            prompt = f"""Extract ALL specific tool/software names mentioned in this refined query.
Return ONLY actual tool names, not generic terms.

Refined Query:
{refined_query}

Rules:
- Return ONLY specific tool names (e.g., "Workday", "ADP", "BambooHR", "Slack")
- Do NOT return generic terms (e.g., "payroll tool", "HR software")
- Look for tools mentioned in sections like "Tools:", "Current Setup:", "Integrations:", etc.
- If no specific tools mentioned, return empty list
- Return as JSON array: ["Tool1", "Tool2"]

Examples:
Query: "Tools: Workday, ADP, BambooHR"
Response: ["Workday", "ADP", "BambooHR"]

Query: "Current Setup: Slack, Trello, Asana"
Response: ["Slack", "Trello", "Asana"]

Now extract from the refined query above. Return ONLY the JSON array, no explanations."""

            llm = get_shared_llm()
            response = await llm.generate_response(prompt)

            # Parse JSON response
            import json
            import re

            json_match = re.search(r"\[.*?\]", response, re.DOTALL)
            if json_match:
                tools = json.loads(json_match.group())
                logger.info(f"✅ Extracted tools from refined query: {tools}")
                return tools if isinstance(tools, list) else []

            logger.info("ℹ️ No tools extracted from refined query")
            return []

        except Exception as e:
            logger.error(f"Error extracting tools from refined query: {e}")
            return []

    async def _smart_search_for_tools(
        self,
        query: str,
        tool_names: list,
        refined_query: str = None,
        max_results: int = 10,
    ) -> list:
        """Two-phase smart search: exact match first, then semantic.

        Args:
            query: Original search query
            tool_names: List of extracted tool names (if any)
            refined_query: Refined query text (for filtering)
            max_results: Maximum number of results

        Returns:
            List of found tools
        """
        try:
            tools = []
            refined_tools = []

            # Extract tools from refined query if provided
            if refined_query:
                refined_tools = await self._extract_tools_from_refined_query(
                    refined_query
                )
                logger.info(
                    f"📋 Extracted {len(refined_tools)} tools from refined query: {refined_tools}"
                )

            # PHASE 1: If specific tool names extracted → exact match strategy
            if tool_names:
                logger.info(f"🎯 PHASE 1: Exact match search for: {tool_names}")

                # Step 1a: Search Pinecone with exact name matching
                logger.info("🔍 Step 1a: Searching Pinecone for exact matches...")
                from ai_tool_recommender.ai_agents.tools.pinecone import PineconeService

                pinecone_service = PineconeService()
                pinecone_results = await pinecone_service.search_tools_exact_match(
                    tool_names, max_results=max_results
                )
                pinecone_count = len(pinecone_results) if pinecone_results else 0

                # Track seen tools to prevent duplicates
                seen_tool_identifiers = set()

                if pinecone_results:
                    for tool in pinecone_results:
                        tool_identifier = self._create_node_identifier(
                            tool.get("Title", ""), tool.get("Website", "")
                        )
                        if tool_identifier not in seen_tool_identifiers:
                            tools.append(tool)
                            seen_tool_identifiers.add(tool_identifier)
                            logger.info(
                                f"✅ [PINECONE] Added: {tool.get('Title', 'Unknown')}"
                            )
                        else:
                            logger.warning(
                                f"🔄 [DUPLICATE] Skipped Pinecone duplicate: {tool.get('Title', 'Unknown')}"
                            )

                    logger.info(
                        f"✅ Found {len(tools)} unique exact matches in Pinecone (removed {pinecone_count - len(tools)} duplicates)"
                    )

                # Step 1b: Check which tools were found
                found_tool_names = [t.get("Title", "").strip() for t in tools]
                missing_tools = [
                    name
                    for name in tool_names
                    if not any(
                        self._normalize_tool_name(name)
                        == self._normalize_tool_name(found)
                        for found in found_tool_names
                    )
                ]

                # Step 1c: If not all found, search Internet for missing tools
                if missing_tools:
                    logger.info(
                        f"🔍 Step 1c: Searching Internet for missing tools: {missing_tools}"
                    )
                    from ai_tool_recommender.ai_agents.tools.internet_search import (
                        InternetSearchService,
                    )

                    internet_service = InternetSearchService()
                    internet_results = await internet_service.search_ai_tools_exact(
                        missing_tools, max_results=max_results
                    )

                    if internet_results:
                        # Add ALL internet results (deduplication disabled per user request)
                        for idx, tool in enumerate(internet_results, 1):
                            # Always add the tool, no deduplication check
                            tools.append(tool)
                        logger.info(
                            f"✅ [INTERNET] Added: {tool.get('Title', 'Unknown')} (deduplication disabled)"
                        )

                        logger.info(
                            f"✅ Added ALL {len(internet_results)} tools from Internet (deduplication disabled)"
                        )

                        # Re-check which tools were found after Internet search
                        found_tool_names_after_internet = [
                            t.get("Title", "").strip() for t in tools
                        ]
                        still_missing = [
                            name
                            for name in tool_names
                            if not any(
                                self._normalize_tool_name(name)
                                == self._normalize_tool_name(found)
                                for found in found_tool_names_after_internet
                            )
                        ]

                        if not still_missing:
                            logger.info(
                                f"✅ All requested tools found! Pinecone: {pinecone_count}, Internet: {len(internet_results)}"
                            )
                            # Return immediately if all tools found (skip refined query filtering and semantic fallback)
                            return tools[:max_results]

                # Step 1d: Filter by refined query tools if exists
                # BUT: Only filter if we have more tools than requested (don't filter out the exact tool we're looking for)
                if refined_tools and tools and len(tools) > len(tool_names):
                    logger.info(
                        f"🔍 Step 1d: Filtering by refined query tools: {refined_tools}"
                    )
                    filtered_tools = []
                    for tool in tools:
                        tool_title = tool.get("Title", "").strip()
                        # Check if tool matches any refined query tool
                        matches_refined = any(
                            self._normalize_tool_name(tool_title)
                            == self._normalize_tool_name(refined_tool)
                            or self._normalize_tool_name(refined_tool)
                            in self._normalize_tool_name(tool_title)
                            for refined_tool in refined_tools
                        )
                        # Also keep tools that match the requested tool names (don't filter out what user asked for)
                        matches_requested = any(
                            self._normalize_tool_name(tool_title)
                            == self._normalize_tool_name(requested_tool)
                            for requested_tool in tool_names
                        )
                        if matches_refined or matches_requested:
                            filtered_tools.append(tool)

                    if filtered_tools:
                        tools = filtered_tools
                        logger.info(
                            f"✅ Filtered to {len(tools)} tools matching refined query or requested tools"
                        )

                # Step 1e: If no exact matches found, try semantic search as fallback
                # but ONLY return tools that contain the tool name in their title
                if not tools and tool_names:
                    logger.info("=" * 80)
                    logger.info(
                        f"⚠️ [FALLBACK] No exact matches found for {tool_names}"
                    )
                    logger.info(
                        "🌐 [FALLBACK] Trying semantic search with Gemini internet search..."
                    )
                    logger.info("=" * 80)
                    search_result = await self.recommender.search_tools(
                        query=query,
                        max_results=max_results * 3,  # Get more results for filtering
                        include_pinecone=True,
                        include_internet=True,  # ✅ This triggers Gemini search!
                    )

                    if search_result.get("status") == "success" and search_result.get(
                        "tools"
                    ):
                        semantic_tools = search_result.get("tools", [])

                        # ONLY return tools that contain the tool name in their title
                        # Don't return unrelated tools
                        matching_tools = []

                        for tool in semantic_tools:
                            tool_title = tool.get("Title", "").strip()
                            if not tool_title:
                                continue

                            tool_title_lower = self._normalize_tool_name(tool_title)

                            # Check if any tool name is in the tool title (exact or partial)
                            matches_name = any(
                                self._normalize_tool_name(tool_name)
                                == tool_title_lower  # Exact match
                                or self._normalize_tool_name(tool_name)
                                in tool_title_lower  # Tool name in title
                                or tool_title_lower
                                in self._normalize_tool_name(
                                    tool_name
                                )  # Title in tool name
                                for tool_name in tool_names
                            )

                            if matches_name:
                                matching_tools.append(tool)
                                logger.info(
                                    f"✅ Found matching tool in semantic search: {tool_title}"
                                )

                        # ONLY return tools that match the name - don't return unrelated tools
                        if matching_tools:
                            tools = matching_tools[:max_results]
                            logger.info(
                                f"✅ Semantic fallback found {len(matching_tools)} tools matching '{tool_names}'"
                            )
                        else:
                            logger.warning(
                                f"❌ Semantic fallback found no tools matching '{tool_names}' - returning empty"
                            )
                            tools = []  # Return empty if no name matches found

                return tools[:max_results]

            # PHASE 2: If generic query (no specific tool names) → semantic search
            else:
                logger.info("=" * 80)
                logger.info(
                    "🔍 [PHASE 2] Generic tool discovery query - no specific tool names extracted"
                )
                logger.info(f"📝 [PHASE 2] Query: '{query}'")
                logger.info(
                    "🌐 [PHASE 2] Will search BOTH Pinecone AND Gemini internet search"
                )
                logger.info("=" * 80)
                search_result = await self.recommender.search_tools(
                    query=query,
                    max_results=max_results,
                    include_pinecone=True,
                    include_internet=True,  # ✅ This triggers Gemini search!
                )
                logger.info("=" * 80)
                logger.info(
                    f"✅ [PHASE 2] Search completed - status: {search_result.get('status')}"
                )
                logger.info(
                    f"📊 [PHASE 2] Tools found: {len(search_result.get('tools', []))}"
                )
                logger.info("=" * 80)

                if search_result.get("status") == "success" and search_result.get(
                    "tools"
                ):
                    tools = search_result.get("tools", [])

                    # Filter by refined query tools if exists
                    if refined_tools and tools:
                        logger.info(
                            f"🔍 Filtering semantic results by refined query tools: {refined_tools}"
                        )
                        filtered_tools = []
                        for tool in tools:
                            tool_title = tool.get("Title", "").strip()
                            # Check if tool matches any refined query tool
                            matches_refined = any(
                                self._normalize_tool_name(tool_title)
                                == self._normalize_tool_name(refined_tool)
                                or self._normalize_tool_name(refined_tool)
                                in self._normalize_tool_name(tool_title)
                                for refined_tool in refined_tools
                            )
                            if matches_refined:
                                filtered_tools.append(tool)

                        if filtered_tools:
                            tools = filtered_tools
                            logger.info(
                                f"✅ Filtered to {len(tools)} tools matching refined query"
                            )

                return tools[:max_results]

        except Exception as e:
            logger.error(f"Error in smart search: {e}", exc_info=True)
            return []

    async def _auto_generate_workflow(
        self,
        refined_query: str,
        conversation,
        workflow_id: str,
        request_user,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Automatically generate workflow from refined query.

        Args:
            refined_query: The refined query text
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response dictionary with workflow
        """
        try:
            workflow_start_time = time.time()
            logger.info("=" * 100)
            logger.info(
                "🔨 [WORKFLOW GENERATION] STARTING AUTO-GENERATION FROM REFINED QUERY"
            )
            logger.info("=" * 100)
            logger.info(f"📝 [REFINED QUERY] {refined_query[:500]}...")
            logger.info(f"🆔 [WORKFLOW ID] {workflow_id}")
            logger.info("=" * 100)

            # Step 0: Extract tool names from refined query
            tool_extraction_start = time.time()
            logger.info(
                "🔍 [STEP 0] Extracting specific tool names from refined query..."
            )
            tool_names = await self._extract_tool_names_from_query(refined_query)
            tool_extraction_time = time.time() - tool_extraction_start
            logger.info(
                f"✅ [STEP 0] Extracted {len(tool_names)} tool names: {tool_names}"
            )
            logger.info(f"⏱️ [STEP 0] Tool extraction took {tool_extraction_time:.2f}s")

            # Use smart search with refined query filtering
            # Request MORE tools to ensure we get enough from Gemini internet search
            search_start = time.time()
            logger.info("=" * 100)
            logger.info("🔍 [STEP 1] STARTING INTELLIGENT TOOL SEARCH")
            logger.info(f"📝 [SEARCH STRATEGY] Tool names: {tool_names}")
            logger.info(
                f"📊 [SEARCH PARAMS] Max results: {kwargs.get('max_results', 8)}"
            )
            logger.info(
                "🎯 [SEARCH FLOW] Pinecone (priority) → Internet (for missing tools)"
            )
            logger.info("=" * 100)

            tools = await self._smart_search_for_tools(
                query=refined_query,
                tool_names=tool_names,
                refined_query=refined_query,  # Filter by refined query
                max_results=kwargs.get(
                    "max_results", 8
                ),  # Maximum 8 tools from Pinecone + Gemini search combined
            )
            search_time = time.time() - search_start

            logger.info("=" * 100)
            logger.info(f"✅ [STEP 1] TOOL SEARCH COMPLETED")
            logger.info(f"📊 [RESULTS] Found {len(tools)} tools")
            logger.info(f"⏱️ [TIMING] Search took {search_time:.2f}s")

            # Log tool sources breakdown
            pinecone_count = sum(1 for t in tools if "Pinecone" in t.get("Source", ""))
            internet_count = sum(1 for t in tools if "Internet" in t.get("Source", ""))
            logger.info(
                f"📦 [SOURCE BREAKDOWN] Pinecone: {pinecone_count}, Internet: {internet_count}"
            )
            logger.info("=" * 100)

            if not tools:
                return self.format_response(
                    message="I couldn't find any relevant tools for your refined query.",
                    suggestions=[
                        "Try modifying the query",
                        "Be more specific",
                    ],
                )

            # Generate workflow without timeout - unlimited processing time per user request
            workflow = None
            workflow_gen_start = time.time()
            logger.info("=" * 100)
            logger.info("🏗️ [STEP 2] GENERATING WORKFLOW WITH SEQUENTIAL LOGIC")
            logger.info(f"📊 [INPUT] {len(tools)} tools to organize into workflow")
            logger.info(
                "🎯 [STRATEGY] Analyze tool capabilities → Determine execution sequence → Create nodes & edges"
            )
            logger.info("=" * 100)

            try:
                workflow = await self.recommender.generate_workflow(
                    refined_query, tools
                )
                workflow_gen_time = time.time() - workflow_gen_start

                if workflow:
                    nodes_count = len(workflow.get("nodes", []))
                    edges_count = len(workflow.get("edges", []))
                    logger.info("=" * 100)
                    logger.info("✅ [STEP 2] WORKFLOW GENERATION COMPLETED")
                    logger.info(
                        f"📊 [WORKFLOW] {nodes_count} nodes, {edges_count} edges"
                    )
                    logger.info(f"⏱️ [TIMING] Generation took {workflow_gen_time:.2f}s")
                    logger.info("=" * 100)

                    # Log workflow structure for debugging
                    logger.info("📋 [WORKFLOW STRUCTURE]")
                    for i, node in enumerate(
                        workflow.get("nodes", [])[:10], 1
                    ):  # Log first 10 nodes
                        node_data = node.get("data", {})
                        logger.info(
                            f"   {i}. {node_data.get('label', 'Unknown')} - {node_data.get('workflow_stage', 'N/A')}"
                        )
                    if nodes_count > 10:
                        logger.info(f"   ... and {nodes_count - 10} more nodes")
                    logger.info("=" * 100)
            except Exception as e:
                workflow_gen_time = time.time() - workflow_gen_start
                logger.error("=" * 100)
                logger.error(
                    f"❌ [STEP 2] WORKFLOW GENERATION FAILED after {workflow_gen_time:.2f}s"
                )
                logger.error(f"❌ [ERROR] {e}")
                logger.error("=" * 100)

            if not workflow:
                return self.format_response(
                    message="I found tools but couldn't generate a workflow. Please try again.",
                    tools_mentioned=[tool.get("Title") for tool in tools[:5]],
                    suggestions=["Try again", "Simplify query"],
                )

            # Update conversation with workflow
            conversation.workflow_nodes = workflow.get("nodes", [])
            conversation.workflow_edges = workflow.get("edges", [])
            await sync_to_async(conversation.save)()

            logger.info("✅ Workflow auto-generated and saved!")

            # Format tools mentioned
            tools_mentioned = [tool.get("Title") for tool in tools[:10]]

            # Save workflow generation log (non-blocking - don't fail if this errors)
            db_save_start = time.time()
            try:
                from ai_tool_recommender.models import WorkflowGeneration

                await sync_to_async(WorkflowGeneration.objects.create)(
                    id=workflow_id,
                    user=request_user,
                    query=refined_query,
                    workflow_data={
                        "nodes": workflow.get("nodes", []),
                        "edges": workflow.get("edges", []),
                    },
                    tools_count=len(tools),
                    generation_method="llm",
                )
                logger.info("✅ Workflow generation log saved")
            except Exception as log_error:
                logger.warning(
                    f"⚠️ Failed to save workflow generation log (non-critical): {log_error}"
                )

            # Save to Workflow model with Node/Edge tables (non-blocking - don't fail if this errors)
            try:
                from workflow.models import Workflow

                # Create or update Workflow model
                def create_or_update_workflow():
                    return Workflow.objects.update_or_create(
                        id=str(workflow_id),
                        defaults={
                            "name": f"Workflow - {refined_query[:50]}",
                            "owner": request_user,
                            "user_query": refined_query,
                            "prompt": refined_query,
                            "metadata": {
                                "nodes": workflow.get("nodes", []),
                                "edges": workflow.get("edges", []),
                            },
                        },
                    )

                workflow_obj, created = await sync_to_async(create_or_update_workflow)()
                logger.info(
                    f"✅ {'Created' if created else 'Updated'} Workflow model: {workflow_id}"
                )

                # Save nodes and edges to separate Node/Edge tables
                workflow_service = WorkflowGenerationService()
                await workflow_service._save_to_node_edge_tables(workflow_obj, workflow)
                logger.info(
                    f"✅ Saved {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges to Node/Edge tables"
                )
            except Exception as workflow_error:
                logger.warning(
                    f"⚠️ Failed to save to Workflow/Node/Edge tables (non-critical): {workflow_error}"
                )

            db_save_time = time.time() - db_save_start
            logger.info(f"⏱️ Database save took {db_save_time:.2f}s")

            # Log total workflow generation time
            total_time = time.time() - workflow_start_time
            logger.info(
                f"🎯 TOTAL WORKFLOW GENERATION TIME: {total_time:.2f}s ({int(total_time * 1000)}ms) | "
                f"Breakdown: Extraction={tool_extraction_time:.2f}s, Search={search_time:.2f}s, "
                f"Generation={workflow_gen_time:.2f}s, DB={db_save_time:.2f}s"
            )

            # Background scraping disabled per user request
            # # Add background task to add new internet search tools to Pinecone (non-blocking)
            # if tools:
            #     try:
            #         internet_tools = [
            #             tool
            #             for tool in tools
            #             if "Internet Search" in tool.get("Source", "")
            #         ]
            #
            #         if internet_tools:
            #             logger.info(
            #                 f"🌐 Queuing {len(internet_tools)} internet tools for background processing"
            #             )
            #             task_id = add_background_task(
            #                 background_add_new_tools_to_pinecone,
            #                 internet_tools,
            #                 refined_query,
            #             )
            #             logger.info(
            #                 f"✅ Background task {task_id} queued for Pinecone addition"
            #             )
            #     except Exception as bg_error:
            #         logger.warning(
            #             f"⚠️ Failed to queue background task (non-critical): {bg_error}"
            #         )
            logger.info(
                "ℹ️ Background scraping disabled - internet tools will not be queued for Pinecone addition"
            )

            return self.format_response(
                message=(
                    f"I've generated a workflow with {len(workflow.get('nodes', []))} tools based on your refined query. "
                    "You can now modify it, add more tools, or get implementation guidance."
                ),
                tools_mentioned=tools_mentioned,
                workflow_changes={
                    "nodes_added": len(workflow.get("nodes", [])),
                    "edges_added": len(workflow.get("edges", [])),
                },
                suggestions=[],
                workflow=workflow,
            )

        except Exception as e:
            logger.error(f"❌ Error auto-generating workflow: {e}", exc_info=True)

            # Check if workflow was actually saved despite the error
            # Reload conversation to get latest state
            await sync_to_async(conversation.refresh_from_db)()

            if conversation.workflow_nodes and len(conversation.workflow_nodes) > 0:
                # Workflow was successfully saved, return success message
                logger.info(
                    f"✅ Workflow was saved despite error ({len(conversation.workflow_nodes)} nodes), returning success"
                )
                tools_mentioned = []
                try:
                    # Try to get tools from search result if available
                    search_result = await self.recommender.search_tools(
                        query=refined_query,
                        max_results=10,
                        include_pinecone=kwargs.get("include_pinecone", True),
                        include_internet=kwargs.get("include_internet", True),
                    )
                    tools = search_result.get("tools", [])
                    tools_mentioned = [tool.get("Title") for tool in tools[:10]]
                except Exception:
                    pass  # Ignore errors when getting tools for display

                return self.format_response(
                    message=(
                        f"I've generated a workflow with {len(conversation.workflow_nodes)} tools based on your refined query. "
                        "You can now modify it, add more tools, or get implementation guidance."
                    ),
                    tools_mentioned=tools_mentioned,
                    workflow_changes={
                        "nodes_added": len(conversation.workflow_nodes),
                        "edges_added": len(conversation.workflow_edges),
                    },
                    suggestions=[],
                )
            else:
                # Workflow was not saved, return error
                return self.format_response(
                    message="I saved your refined query, but encountered an error generating the workflow. You can generate it manually.",
                    suggestions=["Generate workflow", "Show me the refined query"],
                )

    def _format_question_message(
        self, question: Dict[str, Any], questionnaire: Dict[str, Any]
    ) -> str:
        """
        Format a question message for the user.

        Args:
            question: Question dictionary
            questionnaire: Full questionnaire state

        Returns:
            Formatted message string
        """
        try:
            # Get progress info
            progress_msg, _ = self.questionnaire_service.get_progress_message(
                questionnaire
            )

            # Check if this is Phase 4 (refinement)
            current_phase = questionnaire.get("current_phase", 1)
            is_refinement = current_phase == 4

            if is_refinement:
                # For Phase 4, include the refined query in the message
                phase_data = questionnaire.get("phase_4", {})
                refined_query = phase_data.get("refined_query", "")

                if refined_query:
                    message = (
                        f"**{progress_msg}**\n\n"
                        f"Based on our conversation, here's what I understand:\n\n"
                        f"{refined_query}\n\n"
                        f"{question['question']}"
                    )
                else:
                    message = f"**{progress_msg}**\n\n{question['question']}"
            else:
                # Regular question
                reason = question.get("reason", "")
                message = f"**{progress_msg}**\n\n{question['question']}"

                if reason:
                    message += f"\n\n*{reason}*"

            return message

        except Exception as e:
            logger.error(f"Error formatting question: {e}")
            return question.get("question", "What would you like to build?")

    async def _handle_workflow_ready_state(
        self, user_message: str, conversation, workflow_id: str, request_user, **kwargs
    ) -> Dict[str, Any]:
        """
        Handle all workflow conversations after questionnaire completion.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response dictionary
        """
        try:
            current_state = conversation.workflow_state

            # Check if this is a workflow generation request
            if await self._is_workflow_generation_request(user_message, current_state):
                # Get refined query
                refined_query_obj = await self._get_refined_query(workflow_id)
                query = (
                    refined_query_obj.refined_query
                    if refined_query_obj
                    else user_message
                )
                return await self._auto_generate_workflow(
                    refined_query=query,
                    conversation=conversation,
                    workflow_id=workflow_id,
                    request_user=request_user,
                    **kwargs,
                )

            # Check if this is a tool modification request (add/remove/modify)
            intent_analysis = await self.conversation_ai.analyze_user_intent(
                user_message, conversation.chat_history, current_state
            )

            intent = intent_analysis.get("intent")

            if intent == "add_tool":
                return await self._handle_add_tool(
                    user_message, conversation, workflow_id, request_user
                )
            elif intent == "delete_tool":
                return await self._handle_delete_tool(
                    user_message, conversation, workflow_id, request_user
                )
            elif intent == "tool_inquiry":
                return await self._handle_tool_inquiry(user_message, conversation)
            elif intent == "workflow_question":
                return await self._handle_workflow_question(user_message, conversation)
            elif intent == "explore_tools":
                return await self._handle_explore_tools(user_message, conversation)
            else:
                # General conversation
                return await self._handle_general_conversation(
                    user_message, conversation
                )

        except Exception as e:
            logger.error(
                f"❌ Error in {self.agent_name}.process_message (workflow_ready): {e}",
                exc_info=True,
            )
            return self.format_response(
                message="I encountered an error processing your request. Please try again.",
                suggestions=[],
            )

    async def _is_workflow_generation_request(
        self, user_message: str, current_state: str
    ) -> bool:
        """Check if message is requesting workflow generation."""
        message_lower = user_message.lower()

        generation_keywords = [
            "generate workflow",
            "create workflow",
            "build workflow",
            "make workflow",
            "generate my workflow",
            "create my workflow",
        ]

        return any(keyword in message_lower for keyword in generation_keywords)

    async def _get_refined_query(self, workflow_id: str):
        """Get refined query from database if exists."""
        try:
            refined_query_obj = await sync_to_async(
                lambda: RefinedQuery.objects.filter(workflow_id=workflow_id).first()
            )()
            return refined_query_obj
        except Exception as e:
            logger.error(f"Error fetching refined query: {e}")
            return None

    async def _handle_add_tool(
        self, user_message: str, conversation, workflow_id: str, request_user
    ) -> Dict[str, Any]:
        """Handle adding a tool to workflow - directly adds without asking."""
        try:
            # Extract tool name/query from message
            tool_query = (
                user_message.replace("add", "")
                .replace("include", "")
                .replace("to my workflow", "")
                .replace("to workflow", "")
                .strip()
            )

            # Step 0: Use LLM to identify specific tool names from query
            tool_names = await self._extract_tool_names_from_query(tool_query)

            # Get refined query if exists (for filtering)
            refined_query_text = None
            try:
                refined_query_obj = await self._get_refined_query(workflow_id)
                if refined_query_obj:
                    refined_query_text = refined_query_obj.refined_query
                    logger.info(
                        f"📋 Found refined query for filtering: {refined_query_text[:100]}..."
                    )
            except Exception as e:
                logger.warning(f"Could not get refined query: {e}")

            # Use smart search with exact matching for specific tool names
            tools = await self._smart_search_for_tools(
                query=tool_query,
                tool_names=tool_names,
                refined_query=refined_query_text,
                max_results=5,
            )

            if not tools:
                return self.format_response(
                    message=f"I couldn't find a tool matching '{tool_query}'. Try being more specific.",
                    suggestions=[],
                )

            # Get existing tool identifiers to check for duplicates (comprehensive check)
            existing_identifiers = set()
            for node in conversation.workflow_nodes:
                node_data = node.get("data", {})
                tool_name = node_data.get("label", "").strip()
                website = node_data.get("website", "").strip()
                # Create identifier using same method as workflow generation
                identifier = self._create_node_identifier(tool_name, website)
                existing_identifiers.add(identifier)
                # Also add normalized name for additional check
                if tool_name:
                    normalized = self._normalize_tool_name(tool_name)
                    if normalized:
                        existing_identifiers.add(f"name:{normalized}")

            # Find first non-duplicate tool
            tool_to_add = None
            for tool in tools:
                tool_title = tool.get("Title") or tool.get("title") or ""
                tool_website = tool.get("Website") or tool.get("website") or ""

                # Create identifier for this tool
                tool_identifier = self._create_node_identifier(tool_title, tool_website)

                # Check if duplicate
                if tool_identifier in existing_identifiers:
                    logger.info(
                        f"🔄 Skipping duplicate tool: '{tool_title}' (identifier: {tool_identifier})"
                    )
                    continue

                # Also check normalized name
                if tool_title:
                    normalized = self._normalize_tool_name(tool_title)
                    if normalized and f"name:{normalized}" in existing_identifiers:
                        logger.info(
                            f"🔄 Skipping duplicate tool by name: '{tool_title}'"
                        )
                        continue

                    # Found non-duplicate tool
                    tool_to_add = tool
                    break

            if not tool_to_add:
                return self.format_response(
                    message=f"All found tools are already in your workflow. Try searching for a different tool.",
                    suggestions=[],
                )

            # Get refined query for tags field (if available)
            # Use the refined_query_text we already fetched above, or fallback to original_query
            if not refined_query_text:
                refined_query_text = conversation.original_query or ""

            # Create node from tool
            node_data = self._create_tool_node(
                tool_to_add,
                len(conversation.workflow_nodes) + 1,
                refined_query_text=refined_query_text,
            )

            # Add node to workflow
            conversation.workflow_nodes.append(node_data)

            # Regenerate edges intelligently
            if len(conversation.workflow_nodes) > 1:
                workflow_service = WorkflowGenerationService()
                conversation.workflow_edges = (
                    await workflow_service.regenerate_edges_intelligently(
                        conversation.workflow_nodes,
                        conversation.original_query,
                        workflow_id=workflow_id,
                    )
                )

            # Save conversation
            await sync_to_async(conversation.save)()

            # Create workflow dict for response
            workflow = {
                "nodes": conversation.workflow_nodes,
                "edges": conversation.workflow_edges,
            }

            # Save workflow to database (Workflow/Node/Edge tables) - non-blocking
            try:
                from workflow.models import Workflow

                # Get the query to use for saving
                query = (
                    refined_query_text
                    if refined_query_text
                    else conversation.original_query or "Workflow"
                )

                # Update or create workflow
                workflow_obj, created = await sync_to_async(
                    Workflow.objects.update_or_create
                )(
                    id=str(workflow_id),
                    defaults={
                        "metadata": workflow,
                        "owner": request_user
                        if request_user.is_authenticated
                        else None,
                        "user_query": query,
                        "prompt": query,
                    },
                )

                action = "created" if created else "updated"
                logger.info(f"✅ Workflow {action} with ID: {workflow_id}")

                # Save nodes and edges to separate Node/Edge tables
                try:
                    workflow_service = WorkflowGenerationService()
                    await workflow_service._save_to_node_edge_tables(
                        workflow_obj, workflow
                    )
                    logger.info(
                        f"✅ Saved {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges to Node/Edge tables"
                    )
                except Exception as node_edge_error:
                    logger.warning(
                        f"⚠️ Failed to save to Node/Edge tables (non-critical): {node_edge_error}"
                    )
            except Exception as workflow_error:
                logger.warning(
                    f"⚠️ Failed to save to Workflow/Node/Edge tables (non-critical): {workflow_error}"
                )

            tool_name = tool_to_add.get("Title") or tool_to_add.get("title") or "tool"
            return self.format_response(
                message=f"I've added {tool_name} to your workflow.",
                tools_mentioned=[tool_name],
                workflow_changes={
                    "action": "added",
                    "tool": tool_name,
                    "nodes_count": len(conversation.workflow_nodes),
                },
                suggestions=[],
                workflow=workflow,
            )

        except Exception as e:
            logger.error(f"Error adding tool: {e}", exc_info=True)
            return self.format_response(
                message="I had trouble adding that tool. Please try again.",
                suggestions=[],
            )

    async def _handle_delete_tool(
        self, user_message: str, conversation, workflow_id: str, request_user
    ) -> Dict[str, Any]:
        """Handle removing a tool from workflow - directly removes without asking."""
        try:
            # Extract tool name from message
            remove_keywords = ["remove", "delete", "drop", "take out", "get rid of"]

            # Find tool name by removing keywords
            tool_name = user_message
            for keyword in remove_keywords:
                tool_name = (
                    tool_name.replace(keyword, "")
                    .replace("from my workflow", "")
                    .replace("from workflow", "")
                    .strip()
                )

            if not tool_name:
                return self.format_response(
                    message="Please specify which tool you'd like to remove.",
                    suggestions=[],
                )

            # Find tool in workflow nodes (case-insensitive partial match)
            tool_name_lower = tool_name.lower().strip()
            node_to_remove = None
            node_index = -1

            for i, node in enumerate(conversation.workflow_nodes):
                node_label = node.get("data", {}).get("label", "").lower().strip()
                if tool_name_lower in node_label or node_label in tool_name_lower:
                    node_to_remove = node
                    node_index = i
                    break

            if not node_to_remove:
                return self.format_response(
                    message=f"I couldn't find '{tool_name}' in your workflow. Please check the tool name.",
                    suggestions=[],
                )

            removed_tool_name = node_to_remove.get("data", {}).get("label", "tool")

            # Remove node
            conversation.workflow_nodes.pop(node_index)

            # Remove edges connected to this node
            node_id = node_to_remove.get("id")
            conversation.workflow_edges = [
                edge
                for edge in conversation.workflow_edges
                if edge.get("source") != node_id and edge.get("target") != node_id
            ]

            # Regenerate edges if there are remaining nodes
            if len(conversation.workflow_nodes) > 1:
                workflow_service = WorkflowGenerationService()
                conversation.workflow_edges = (
                    await workflow_service.regenerate_edges_intelligently(
                        conversation.workflow_nodes,
                        conversation.original_query,
                        workflow_id=workflow_id,
                    )
                )
            elif len(conversation.workflow_nodes) == 0:
                conversation.workflow_edges = []

            # Save conversation
            await sync_to_async(conversation.save)()

            # Create workflow dict for response
            workflow = {
                "nodes": conversation.workflow_nodes,
                "edges": conversation.workflow_edges,
            }

            # Save workflow to database (Workflow/Node/Edge tables) - non-blocking
            try:
                from workflow.models import Workflow

                # Get the query to use for saving
                refined_query_text = None
                try:
                    refined_query_obj = await self._get_refined_query(workflow_id)
                    if refined_query_obj:
                        refined_query_text = refined_query_obj.refined_query
                except Exception:
                    pass

                query = (
                    refined_query_text
                    if refined_query_text
                    else conversation.original_query or "Workflow"
                )

                # Update or create workflow
                workflow_obj, created = await sync_to_async(
                    Workflow.objects.update_or_create
                )(
                    id=str(workflow_id),
                    defaults={
                        "metadata": workflow,
                        "owner": request_user
                        if request_user.is_authenticated
                        else None,
                        "user_query": query,
                        "prompt": query,
                    },
                )

                action = "created" if created else "updated"
                logger.info(f"✅ Workflow {action} with ID: {workflow_id}")

                # Save nodes and edges to separate Node/Edge tables
                try:
                    workflow_service = WorkflowGenerationService()
                    await workflow_service._save_to_node_edge_tables(
                        workflow_obj, workflow
                    )
                    logger.info(
                        f"✅ Saved {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges to Node/Edge tables"
                    )
                except Exception as node_edge_error:
                    logger.warning(
                        f"⚠️ Failed to save to Node/Edge tables (non-critical): {node_edge_error}"
                    )
            except Exception as workflow_error:
                logger.warning(
                    f"⚠️ Failed to save to Workflow/Node/Edge tables (non-critical): {workflow_error}"
                )

            return self.format_response(
                message=f"I've removed {removed_tool_name} from your workflow.",
                tools_mentioned=[removed_tool_name],
                workflow_changes={
                    "action": "removed",
                    "tool": removed_tool_name,
                    "nodes_count": len(conversation.workflow_nodes),
                },
                suggestions=[],
                workflow=workflow,
            )

        except Exception as e:
            logger.error(f"Error removing tool: {e}", exc_info=True)
            return self.format_response(
                message="I had trouble removing that tool. Please try again.",
                suggestions=[],
            )

    async def _handle_tool_inquiry(
        self, user_message: str, conversation
    ) -> Dict[str, Any]:
        """Handle questions about tools."""
        try:
            # Search for tools based on inquiry
            search_result = await self.recommender.search_tools(
                query=user_message,
                max_results=5,
                include_pinecone=True,
                include_internet=True,
            )

            tools = search_result.get("tools", [])

            if not tools:
                return self.format_response(
                    message="I couldn't find tools matching your inquiry.",
                    suggestions=[],
                )

            # Format response using conversation AI
            response = await self.conversation_ai.format_tool_inquiry_response(
                tools, user_message
            )

            return self.format_response(
                message=response.get("message"),
                tools_mentioned=[tool.get("Title") for tool in tools],
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error handling tool inquiry: {e}", exc_info=True)
            return self.format_response(
                message="I had trouble finding tools. Please try again.",
                suggestions=[],
            )

    async def _handle_workflow_question(
        self, user_message: str, conversation
    ) -> Dict[str, Any]:
        """Handle questions about the workflow."""
        try:
            # Use conversation AI to answer workflow questions
            answer = await self.conversation_ai.answer_workflow_question(
                user_question=user_message,
                workflow_nodes=conversation.workflow_nodes,
                workflow_edges=conversation.workflow_edges,
                original_query=conversation.original_query,
            )

            return self.format_response(
                message=answer,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error answering workflow question: {e}", exc_info=True)
            return self.format_response(
                message="I can explain your workflow. What would you like to know?",
                suggestions=[],
            )

    async def _handle_explore_tools(
        self, user_message: str, conversation
    ) -> Dict[str, Any]:
        """Handle tool exploration requests."""
        try:
            # Search for tools
            search_result = await self.recommender.search_tools(
                query=user_message,
                max_results=8,
                include_pinecone=True,
                include_internet=True,
            )

            tools = search_result.get("tools", [])

            if not tools:
                return self.format_response(
                    message="I couldn't find tools for that request.",
                    suggestions=[],
                )

            # Format response
            response = await self.conversation_ai.format_tools_response(
                tools, user_message
            )

            return self.format_response(
                message=response.get("message"),
                tools_mentioned=[tool.get("Title") for tool in tools],
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error exploring tools: {e}", exc_info=True)
            return self.format_response(
                message="I had trouble finding tools. Please try again.",
                suggestions=[],
            )

    async def _handle_general_conversation(
        self, user_message: str, conversation
    ) -> Dict[str, Any]:
        """Handle general conversation about workflow."""
        try:
            # Use conversation AI for general chat
            response = await self.conversation_ai.generate_conversational_response(
                message=user_message,
                intent="general_chat",
                context={
                    "state": conversation.workflow_state,
                    "has_workflow": bool(conversation.workflow_nodes),
                    "workflow_nodes": conversation.workflow_nodes,
                    "chat_history": conversation.chat_history[-3:]
                    if conversation.chat_history
                    else [],
                },
            )

            return self.format_response(
                message=response,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Error in general conversation: {e}", exc_info=True)
            return self.format_response(
                message="I'm here to help with your workflow. What would you like to do?",
                suggestions=[],
            )

    def _create_tool_node(
        self, tool: Dict[str, Any], sequence_num: int, refined_query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Create a standardized tool node from tool data.

        Args:
            tool: Tool dictionary with Title, Description, Features, etc.
            sequence_num: Sequence number for this node (1-based)
            refined_query_text: Refined query text to use in tags field (optional)

        Returns:
            Node dictionary with id, type, data, and position
        """
        node_id = str(uuid.uuid4())

        # Extract and clean description
        raw_description = tool.get("Description") or tool.get("description") or ""

        # Extract and normalize features
        features = tool.get("Features") or tool.get("features") or []
        if isinstance(features, str):
            features = [f.strip() for f in features.split(",") if f.strip()]

        # Use refined query text for tags (matching existing workflow structure)
        # If no refined query provided, use empty string
        tags = refined_query_text or ""

        # Get tool name
        tool_name = tool.get("Title") or tool.get("title") or tool.get("name") or "Tool"

        # Generate simple recommendation reason
        recommendation_reason = (
            f"Added to workflow for its capabilities in {tool_name.lower()}"
        )

        # Validate website URL
        website = tool.get("Website") or tool.get("website") or ""
        if website and not (
            website.startswith("http://") or website.startswith("https://")
        ):
            website = ""

        return {
            "id": node_id,
            "type": "tool",
            "data": {
                "label": tool_name,
                "description": raw_description,
                "features": features
                if isinstance(features, list)
                else [features]
                if features
                else [],
                "tags": tags,  # Tags is a string (refined query) in existing workflows
                "recommendation_reason": recommendation_reason,
                "website": website,
                "twitter": tool.get("Twitter") or tool.get("twitter") or "",
                "facebook": tool.get("Facebook") or tool.get("facebook") or "",
                "linkedin": tool.get("LinkedIn") or tool.get("linkedin") or "",
                "instagram": tool.get("Instagram") or tool.get("instagram") or "",
                "source": tool.get("Source") or "Pinecone Vector Database",
                "sequence": sequence_num,
            },
            "position": {
                "x": 100 + ((sequence_num - 1) % 3) * 250,
                "y": 100 + ((sequence_num - 1) // 3) * 150,
            },
        }

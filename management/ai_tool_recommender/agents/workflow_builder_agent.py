"""Workflow Builder Agent - Handles workflow generation from refined queries."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from asgiref.sync import sync_to_async

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.tools.ai_tool_recommender import AIToolRecommender
from ai_tool_recommender.conversational_service import ConversationAI
from ai_tool_recommender.models import RefinedQuery
from ai_tool_recommender.workflow_generation_service import WorkflowGenerationService

logger = logging.getLogger(__name__)


class WorkflowBuilderAgent(BaseAgent):
    """
    Agent that handles workflow generation from refined queries.

    This agent:
    - Takes a refined query (single or detailed)
    - Searches for relevant tools
    - Generates workflow with found tools
    - Handles conversational workflow modifications
    - Can add/remove/modify nodes
    """

    def __init__(self):
        """Initialize the workflow builder agent."""
        super().__init__()
        self.recommender = AIToolRecommender()
        self.conversation_ai = ConversationAI()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "workflow_builder"

    async def can_handle(
        self, user_message: str, conversation, current_state: str, **kwargs
    ) -> bool:
        """
        Check if this agent should handle the message.

        This agent handles:
        - workflow_ready state (modifying workflows)
        - Direct workflow generation requests
        - Tool addition/removal

        Args:
            user_message: User's message
            conversation: ConversationSession object
            current_state: Current workflow state
            **kwargs: Additional parameters (max_results, include_pinecone, etc.)

        Returns:
            True if agent can handle
        """
        # Handle workflow_ready state
        if current_state == "workflow_ready":
            return True

        # Check if message is a direct workflow generation request
        # (single query without going through questionnaire)
        message_lower = user_message.lower()
        direct_workflow_keywords = [
            "generate workflow",
            "create workflow",
            "build workflow",
            "make workflow",
        ]

        if any(keyword in message_lower for keyword in direct_workflow_keywords):
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
        Process user message for workflow building.

        Args:
            user_message: User's message
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters (max_results, include_pinecone, etc.)

        Returns:
            Response dictionary
        """
        try:
            current_state = conversation.workflow_state

            logger.info(
                f"🤖 {self.agent_name} processing message in state: {current_state}"
            )

            # Check if this is a simple greeting - redirect to general_assistant
            message_lower = user_message.lower().strip()
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

            # Check if refined query exists
            refined_query_obj = await self._get_refined_query(workflow_id)

            # If no refined query, proceed directly with the query (no suggestions)
            # Just use the user message as the query

            # Determine the query to use
            if refined_query_obj:
                # Use refined query from database
                query = refined_query_obj.refined_query
                logger.info(f"📝 Using refined query from database: {query[:100]}...")
            else:
                # Use user message as query
                query = user_message
                logger.info(f"📝 Using user message as query: {query[:100]}...")

            # Always generate workflow directly (no questions, no suggestions)
            # If workflow already exists and user is asking something else, check intent
            if current_state == "workflow_ready" and conversation.workflow_nodes:
                # Workflow exists, check if user wants to modify or just chat
                pass  # Continue to intent analysis below
            else:
                # No workflow or initial state - generate workflow directly
                return await self._generate_workflow(
                    query, conversation, workflow_id, request_user, **kwargs
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
                f"❌ Error in {self.agent_name}.process_message: {e}", exc_info=True
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

    async def _generate_workflow(
        self,
        query: str,
        conversation,
        workflow_id: str,
        request_user,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate workflow from query.

        Args:
            query: Search query (refined or direct)
            conversation: ConversationSession object
            workflow_id: Workflow UUID
            request_user: User object
            **kwargs: Additional parameters

        Returns:
            Response with generated workflow
        """
        try:
            workflow_start_time = time.time()
            logger.info("=" * 100)
            logger.info("🔨 [WORKFLOW GENERATION] STARTING DIRECT WORKFLOW GENERATION")
            logger.info("=" * 100)
            logger.info(f"📝 [QUERY] {query[:500]}...")
            logger.info(f"🆔 [WORKFLOW ID] {workflow_id}")
            logger.info("=" * 100)

            # First, try to identify specific tool names from the query using LLM
            tool_extraction_start = time.time()
            logger.info("🔍 [STEP 0] Extracting specific tool names from query...")
            tool_names = await self._extract_tool_names_from_query(query)
            tool_extraction_time = time.time() - tool_extraction_start
            logger.info(
                f"✅ [STEP 0] Extracted {len(tool_names)} tool names: {tool_names}"
            )
            logger.info(f"⏱️ [STEP 0] Tool extraction took {tool_extraction_time:.2f}s")

            # Get refined query if this is a refined query (for filtering)
            refined_query_text = query if len(query) > 100 else None

            # Use smart search strategy (exact match for tool names, semantic for generic)
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
                query=query,
                tool_names=tool_names,
                refined_query=refined_query_text,
                original_query=conversation.original_query,  # ✅ Pass original query context
                max_results=kwargs.get("max_results", 8),
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
                    message="I couldn't find any relevant tools for your query.",
                    suggestions=[],
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
                workflow = await self.recommender.generate_workflow(query, tools)
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
                    suggestions=[],
                )

            # Update conversation with workflow
            conversation.workflow_nodes = workflow.get("nodes", [])
            conversation.workflow_edges = workflow.get("edges", [])
            conversation.workflow_state = "workflow_ready"
            await sync_to_async(conversation.save)()

            # Save refined query to database (treat user query as refined query for workflow_builder)
            try:
                from ai_tool_recommender.models import RefinedQuery

                await sync_to_async(RefinedQuery.objects.update_or_create)(
                    workflow_id=workflow_id,
                    defaults={
                        "user": request_user,
                        "session": conversation,
                        "original_query": conversation.original_query,
                        "refined_query": query,  # User query is treated as refined query
                        "workflow_info": {
                            "source": "workflow_builder",
                            "tools_found": len(tools),
                            "workflow_generated": True,
                        },
                    },
                )
                logger.info(
                    f"✅ Saved refined query (user query) to database for workflow {workflow_id}"
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to save refined query to database (non-critical): {e}"
                )

            # Save workflow to database
            db_save_start = time.time()
            await self._save_workflow_to_database(
                workflow, query, workflow_id, request_user, tools
            )
            db_save_time = time.time() - db_save_start
            logger.info(f"⏱️ Database save took {db_save_time:.2f}s")

            # Queue background task for internet tools
            # Background scraping disabled per user request
            # internet_tools = [
            #     tool for tool in tools if "Internet Search" in tool.get("Source", "")
            # ]
            #
            # if internet_tools:
            #     task_id = add_background_task(
            #         background_add_new_tools_to_pinecone, internet_tools, query
            #     )
            #     logger.info(
            #         f"🔄 Background task queued: {task_id} - {len(internet_tools)} tools"
            #     )
            logger.info(
                "ℹ️ Background scraping disabled - internet tools will not be queued for Pinecone addition"
            )

            # Extract tool names from workflow nodes for descriptive message
            tool_names = [
                node.get("data", {}).get("label", "")
                for node in workflow.get("nodes", [])
                if node.get("data", {}).get("label")
            ]
            tools_mentioned = [tool.get("Title") for tool in tools[:10]]

            # Generate comprehensive workflow description using LLM
            description_start = time.time()
            message = await self._generate_workflow_description(
                workflow=workflow,
                user_query=query,
                tools=tools,
            )
            description_time = time.time() - description_start
            logger.info(
                f"⏱️ Workflow description generation took {description_time:.2f}s"
            )

            # Log total workflow generation time (after all operations)
            total_time = time.time() - workflow_start_time
            logger.info(
                f"🎯 TOTAL WORKFLOW GENERATION TIME: {total_time:.2f}s ({int(total_time * 1000)}ms) | "
                f"Breakdown: Extraction={tool_extraction_time:.2f}s, Search={search_time:.2f}s, "
                f"LLM_Generation={workflow_gen_time:.2f}s, Description={description_time:.2f}s, DB={db_save_time:.2f}s"
            )

            return self.format_response(
                message=message,
                tools_mentioned=tools_mentioned,
                workflow_changes={
                    "action": "generated",
                    "nodes_count": len(workflow.get("nodes", [])),
                },
                suggestions=[],
                workflow=workflow,
            )

        except Exception as e:
            logger.error(f"❌ Error generating workflow: {e}", exc_info=True)
            raise

    async def _save_workflow_to_database(
        self, workflow: dict, query: str, workflow_id: str, request_user, tools: list
    ):
        """Save workflow to database."""
        try:
            # Save workflow generation log (non-blocking - don't fail if this errors)
            # This matches refine_query_agent behavior for consistency
            try:
                from ai_tool_recommender.models import WorkflowGeneration

                await sync_to_async(WorkflowGeneration.objects.create)(
                    id=workflow_id,
                    user=request_user if request_user.is_authenticated else None,
                    query=query,
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

                # Save nodes and edges to separate Node/Edge tables (matching refine_query_agent)
                try:
                    from ai_tool_recommender.workflow_generation_service import (
                        WorkflowGenerationService,
                    )

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

            # Save nodes and edges to separate Node/Edge tables (matching refine_query_agent)
            try:
                from ai_tool_recommender.workflow_generation_service import (
                    WorkflowGenerationService,
                )

                workflow_service = WorkflowGenerationService()
                await workflow_service._save_to_node_edge_tables(workflow_obj, workflow)
                logger.info(
                    f"✅ Saved {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges to Node/Edge tables"
                )
            except Exception as node_edge_error:
                logger.warning(
                    f"⚠️ Failed to save to Node/Edge tables (non-critical): {node_edge_error}"
                )

        except Exception as e:
            logger.error(f"❌ Error saving workflow: {e}", exc_info=True)

    async def _generate_workflow_description(
        self, workflow: dict, user_query: str, tools: list
    ) -> str:
        """
        Generate a comprehensive workflow description using LLM.

        Args:
            workflow: The generated workflow with nodes and edges
            user_query: The original user query
            tools: List of tools used in the workflow

        Returns:
            A detailed description of the workflow and how it works
        """
        try:
            # Extract workflow information
            nodes = workflow.get("nodes", [])
            edges = workflow.get("edges", [])

            # Build tool information for the prompt
            tool_info = []
            for i, node in enumerate(nodes, 1):
                node_data = node.get("data", {})
                tool_name = node_data.get("label", "Unknown Tool")
                tool_description = node_data.get("description", "")
                tool_website = node_data.get("website", "")
                recommendation_reason = node_data.get("recommendation_reason", "")

                tool_info.append(
                    {
                        "number": i,
                        "name": tool_name,
                        "description": tool_description,
                        "website": tool_website,
                        "reason": recommendation_reason,
                    }
                )

            # Build edge information (connections)
            edge_info = []
            for edge in edges:
                source_id = edge.get("source", "")
                target_id = edge.get("target", "")
                # Find source and target node names
                source_node = next((n for n in nodes if n.get("id") == source_id), None)
                target_node = next((n for n in nodes if n.get("id") == target_id), None)
                if source_node and target_node:
                    source_name = source_node.get("data", {}).get("label", "")
                    target_name = target_node.get("data", {}).get("label", "")
                    edge_info.append(f"{source_name} → {target_name}")

            # Create comprehensive prompt
            prompt = f"""You are a workflow automation expert. Generate a comprehensive, user-friendly description of this workflow.

USER'S GOAL: {user_query}

WORKFLOW TOOLS:
{chr(10).join([f"{t['number']}. {t['name']} - {t['description']}{chr(10)}   Why it's included: {t['reason']}" for t in tool_info])}

WORKFLOW FLOW:
{chr(10).join(edge_info) if edge_info else "Sequential workflow (tools work in order)"}

Generate a detailed description that includes:
1. **Overview**: What this workflow accomplishes overall
2. **How it works**: Step-by-step explanation of the workflow process
3. **Tool roles**: How each tool contributes to achieving the goal
4. **Data flow**: How information moves between tools
5. **Benefits**: What the user gains from this automation

Write in a clear, conversational tone. Be specific about how the tools work together.
Make it informative but not overly technical. Use markdown formatting for readability.

Description:"""

            # Get LLM instance and generate description
            llm = get_shared_llm()
            description = await llm.generate_response(prompt)

            # Clean up the description
            description = description.strip()

            # If description is too short or seems incomplete, add a fallback
            if len(description) < 100:
                logger.warning(
                    "Generated workflow description seems too short, using fallback"
                )
                tool_names = [t["name"] for t in tool_info]
                description = f"""I've created a workflow to help you: **{user_query}**

**Workflow Overview:**
This workflow uses {len(tool_names)} tools working together to automate your process.

**Tools Included:**
{chr(10).join([f"- **{name}**: {next((t['description'] for t in tool_info if t['name'] == name), 'Helps automate your workflow')}" for name in tool_names])}

**How It Works:**
The tools are connected in a sequence{' → '.join([t['name'] for t in tool_info])} to streamline your workflow. Each tool handles a specific part of the process, passing data to the next tool to complete your automation goal.

**Benefits:**
This automation will save you time and reduce manual work by connecting these tools together."""

            logger.info(
                f"✅ Generated workflow description ({len(description)} characters)"
            )
            return description

        except Exception as e:
            logger.error(f"❌ Error generating workflow description: {e}", exc_info=True)
            # Fallback to simple message if LLM fails
            tool_names = [
                node.get("data", {}).get("label", "")
                for node in nodes
                if node.get("data", {}).get("label")
            ]
            if tool_names:
                if len(tool_names) == 1:
                    return f"I've created a workflow using {tool_names[0]} to help you: {user_query}"
                else:
                    all_but_last = ", ".join(tool_names[:-1])
                    return f"I've created a workflow using {all_but_last}, and {tool_names[-1]} to help you: {user_query}"
            else:
                return f"I've created a workflow with {len(nodes)} tools to help you: {user_query}"

    async def _extract_tools_from_refined_query(self, refined_query: str) -> List[str]:
        """Extract tool names mentioned in refined query.

        Args:
            refined_query: Refined query text (may contain tool names)

        Returns:
            List of extracted tool names
        """
        try:
            if not refined_query:
                return []

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
        tool_names: List[str],
        refined_query: str = None,
        original_query: str = "",
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
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
                        f"✅ Found {len(tools)} unique exact matches in Pinecone (removed {len(pinecone_results) - len(tools)} duplicates)"
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

                logger.info(
                    f"📊 Step 1b: Pinecone search results - Found: {found_tool_names}, Missing: {missing_tools}"
                )

                # Step 1c: If not all found, search Internet for missing tools
                if missing_tools:
                    logger.info(
                        f"✅ Internet search WILL be triggered for: {missing_tools}"
                    )
                    logger.info(
                        f"🔍 Step 1c: Searching Internet for missing tools: {missing_tools}"
                    )
                    from ai_tool_recommender.ai_agents.tools.internet_search import (
                        InternetSearchService,
                    )

                    internet_service = InternetSearchService()
                    logger.info(
                        f"🌐 Calling InternetSearchService.search_ai_tools_exact with: {missing_tools}"
                    )
                    internet_results = await internet_service.search_ai_tools_exact(
                        missing_tools, max_results=max_results
                    )
                    logger.info(
                        f"🌐 InternetSearchService returned {len(internet_results) if internet_results else 0} results"
                    )

                    if not internet_results:
                        logger.warning(
                            f"⚠️ Internet search returned EMPTY results for: {missing_tools}. This might mean:"
                            f" 1) Tool not found on Internet, 2) All results filtered out, 3) Search failed"
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
                                f"✅ All requested tools found! Pinecone: {len(pinecone_results) if pinecone_results else 0}, Internet: {len(internet_results)}"
                            )
                            # Return immediately if all tools found (skip refined query filtering and semantic fallback)
                            return tools[:max_results]
                else:
                    logger.info(
                        f"⏭️ Internet search SKIPPED - All tools found in Pinecone: {found_tool_names}"
                    )

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

                # Step 1e: If no tools found from Pinecone/Internet, try semantic search as fallback
                # If we already have tools from Pinecone, return them (they're already relevant)
                if tools:
                    logger.info(
                        f"✅ Returning {len(tools)} tools found from Pinecone/Internet (skipping semantic fallback)"
                    )
                    return tools[:max_results]

                # Only run semantic fallback if we have no tools yet
                if not tools and tool_names:
                    logger.info("=" * 80)
                    logger.info(
                        f"⚠️ [FALLBACK] No exact matches found for {tool_names}"
                    )
                    logger.info(
                        "🌐 [FALLBACK] Trying semantic search with Gemini internet search..."
                    )
                    logger.info("=" * 80)

                    # Use the original query - let the search engine handle variations
                    search_result = await self.recommender.search_tools(
                        query=query,
                        max_results=max_results * 3,  # Get more results for filtering
                        include_pinecone=True,
                        include_internet=True,
                        use_intelligent_search=True,  # ✅ Ensure intelligent search
                        original_query=original_query,  # ✅ Pass context
                    )

                    if search_result.get("status") == "success" and search_result.get(
                        "tools"
                    ):
                        semantic_tools = search_result.get("tools", [])

                        # ONLY return tools that contain the tool name in their title
                        # Don't return unrelated tools
                        matching_tools = []

                        # Build list of acceptable tool name variations (normalized)
                        acceptable_names = [
                            self._normalize_tool_name(tool_name)
                            for tool_name in tool_names
                        ]

                        for tool in semantic_tools:
                            tool_title = tool.get("Title", "").strip()
                            if not tool_title:
                                continue

                            tool_title_lower = self._normalize_tool_name(tool_title)

                            # Check if any tool name is in the tool title (very flexible matching)
                            # This allows "kfueit" to match "KFUEIT", "Kfueit", or any case variation
                            matches_name = any(
                                acceptable_name
                                == tool_title_lower  # Exact match (case-insensitive)
                                or acceptable_name
                                in tool_title_lower  # Tool name in title (e.g., "excel" in "microsoft excel")
                                or tool_title_lower
                                in acceptable_name  # Title in tool name
                                or tool_title_lower.startswith(
                                    acceptable_name
                                )  # Title starts with tool name
                                or acceptable_name
                                in tool_title_lower.split()  # Tool name is a word in title
                                or (
                                    len(acceptable_name) >= 4
                                    and tool_title_lower.startswith(acceptable_name[:4])
                                )  # First 4 chars match
                                or (
                                    len(acceptable_name) >= 3
                                    and any(
                                        word.startswith(acceptable_name[:3])
                                        for word in tool_title_lower.split()
                                    )
                                )  # Word starts with first 3 chars
                                for acceptable_name in acceptable_names
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
                    max_results=max(
                        max_results, 20
                    ),  # Request at least 20 to get more Gemini tools
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
            logger.info(f"🔍 Extracting tool names from query: '{tool_query}'")
            tool_names = await self._extract_tool_names_from_query(tool_query)
            logger.info(f"✅ Extracted tool names: {tool_names}")

            # Get refined query if exists (for filtering)
            refined_query_text = None
            try:
                from ai_tool_recommender.models import RefinedQuery

                refined_query_obj = await sync_to_async(
                    RefinedQuery.objects.filter(session=conversation)
                    .order_by("-created_at")
                    .first
                )()
                if refined_query_obj:
                    refined_query_text = refined_query_obj.refined_query
                    logger.info(
                        f"📋 Found refined query for filtering: {refined_query_text[:100]}..."
                    )
            except Exception as e:
                logger.warning(f"Could not get refined query: {e}")

            # Use smart search strategy
            tools = await self._smart_search_for_tools(
                query=tool_query,
                tool_names=tool_names,
                refined_query=refined_query_text,
                max_results=5,
            )

            # If still no tools found, return error
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

            # Extract proper tags from tool data (not query text)
            # Use tool's features, description keywords, or category as tags
            tags = await self._extract_tags_from_tool(
                tool_to_add, conversation.original_query
            )

            # Create node from tool
            node_data = self._create_tool_node(
                tool_to_add, len(conversation.workflow_nodes) + 1, tags=tags
            )

            # IMPORTANT: Preserve all existing nodes - append new node to existing workflow
            logger.info(
                f"📋 Current workflow has {len(conversation.workflow_nodes)} nodes before adding new tool"
            )

            # Add node to workflow (preserves all existing nodes)
            conversation.workflow_nodes.append(node_data)

            logger.info(
                f"✅ Added new node. Total nodes now: {len(conversation.workflow_nodes)}"
            )
            logger.info(
                f"📋 All nodes: {[node.get('data', {}).get('label', 'Unknown') for node in conversation.workflow_nodes]}"
            )

            # Regenerate edges intelligently for ALL nodes (preserves existing nodes)
            if len(conversation.workflow_nodes) > 1:
                workflow_service = WorkflowGenerationService()
                conversation.workflow_edges = (
                    await workflow_service.regenerate_edges_intelligently(
                        conversation.workflow_nodes,
                        conversation.original_query,
                        workflow_id=workflow_id,
                    )
                )
                logger.info(
                    f"✅ Regenerated {len(conversation.workflow_edges)} edges for {len(conversation.workflow_nodes)} nodes"
                )

            # Save conversation with ALL nodes (existing + new)
            await sync_to_async(conversation.save)()
            logger.info(
                f"💾 Saved conversation with {len(conversation.workflow_nodes)} total nodes"
            )

            # Create workflow dict for response - includes ALL nodes (existing + new)
            workflow = {
                "nodes": conversation.workflow_nodes,  # All nodes including existing ones
                "edges": conversation.workflow_edges,  # All edges including regenerated ones
            }

            logger.info(
                f"📦 Returning workflow with {len(workflow['nodes'])} nodes and {len(workflow['edges'])} edges"
            )

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
                    from ai_tool_recommender.models import RefinedQuery

                    refined_query_obj = await sync_to_async(
                        RefinedQuery.objects.filter(session=conversation)
                        .order_by("-created_at")
                        .first
                    )()
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
                tools=tools,
                suggestions=response.get("suggestions", []),
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
                tools=tools,
                suggestions=response.get("suggestions", []),
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
        """Handle general conversation."""
        return self.format_response(
            message="I'm here to help you build and modify your workflow. What would you like to do?",
            suggestions=[],
        )

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

    async def _extract_tags_from_tool(
        self, tool: Dict[str, Any], original_query: str
    ) -> str:
        """
        Extract meaningful tags from tool data using LLM.
        Tags should describe what the tool does, not the user's query.

        Args:
            tool: Tool dictionary
            original_query: Original user query for context

        Returns:
            Tags string with meaningful keywords
        """
        try:
            tool_name = tool.get("Title") or tool.get("title") or ""
            description = tool.get("Description") or tool.get("description") or ""
            features = tool.get("Features") or tool.get("features") or []

            if isinstance(features, str):
                features = [f.strip() for f in features.split(",") if f.strip()]

            prompt = f"""Extract meaningful tags/keywords for this tool based on what it actually does.

Tool Name: {tool_name}
Description: {description[:200]}
Features: {', '.join(features[:5]) if features else 'N/A'}

Original User Query (for context only): {original_query}

Generate 3-5 meaningful tags that describe:
1. What the tool does (e.g., "HR automation", "payroll processing")
2. Key capabilities (e.g., "employee management", "time tracking")
3. Industry/use case (e.g., "human resources", "workforce management")

Return ONLY a comma-separated string of tags, no explanations.
Example: "HR automation, payroll processing, employee management, workforce management"

Tags:"""

            llm = get_shared_llm()
            response = await llm.generate_response(prompt)

            # Clean up response - take first line, remove quotes, trim
            tags = response.strip().split("\n")[0].strip().strip('"').strip("'")

            logger.info(f"✅ Extracted tags for {tool_name}: {tags}")
            return tags

        except Exception as e:
            logger.error(f"Error extracting tags: {e}")
            # Fallback: use tool name and basic keywords
            tool_name = tool.get("Title") or tool.get("title") or "tool"
            return f"{tool_name.lower()}, automation, workflow"

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

    def _create_tool_node(
        self, tool: Dict[str, Any], sequence_num: int, tags: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Create a standardized tool node from tool data.

        Args:
            tool: Tool dictionary with Title, Description, Features, etc.
            sequence_num: Sequence number for this node (1-based)
            tags: Proper tags string (extracted from tool, not query text)

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

        # Use provided tags (extracted from tool, not query text)
        # Tags should be meaningful keywords about the tool, not the user's query

        # Get tool name and validate it
        tool_name = tool.get("Title") or tool.get("title") or tool.get("name") or ""

        # Reject tools with garbage names
        garbage_names = [
            "tool",
            "unknown tool",
            "untitled",
            "unknown",
            "",
            "ai tool",
            "software",
        ]
        if not tool_name or tool_name.lower().strip() in garbage_names:
            logger.warning(f"❌ Rejecting tool node with garbage name: '{tool_name}'")
            return None  # Return None to indicate this tool should be skipped

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
                "tags": tags,  # Tags is a string with meaningful keywords about the tool
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

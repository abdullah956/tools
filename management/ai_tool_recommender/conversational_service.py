"""Conversational AI service for interactive workflow building."""

import json
import logging
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.tools.ai_tool_recommender import AIToolRecommender

logger = logging.getLogger(__name__)


class ConversationAI:
    """AI service for handling conversational interactions."""

    def __init__(self):
        """Initialize the conversational AI service."""
        self.recommender = AIToolRecommender()
        self.llm = get_shared_llm()
        # Import questionnaire service for question-based flow
        from .questionnaire_service import QuestionnaireService

        self.questionnaire_service = QuestionnaireService()

    async def analyze_user_intent(
        self, message: str, chat_history: List[Dict], current_state: str = "initial"
    ) -> Dict[str, Any]:
        """
        Analyze user intent from message and chat history.

        Args:
            message: User's message
            chat_history: Conversation history
            current_state: Current workflow state (initial, questioning, workflow_ready, etc.)

        Returns:
            Intent analysis dictionary
        """
        try:
            message_lower = message.lower()

            # Debug logging
            logger.info(f"Analyzing intent for: '{message}' (state: {current_state})")

            # SPECIAL CASE: If in questioning state, treat message as answer
            if current_state == "questioning":
                logger.info("🔒 In questioning state - treating message as answer")
                return {
                    "intent": "answer_question",
                    "confidence": 1.0,
                    "entities": [],
                    "suggested_actions": ["Process answer to current question"],
                }

            # WORKFLOW BUILDING INTENT - Check FIRST (for initial state)
            # Keywords that indicate user wants to BUILD a workflow
            # THIS IS CHECKED FIRST because it's more specific than greetings
            workflow_building_keywords = [
                "i want",
                "i need",
                "help me",
                "build",
                "create",
                "make",
                "automate",
                "workflow for",
                "workflow to",
                "tools for",
                "looking for",
                "trying to",
                "how do i",
                "how can i",
                "set up",
                "setup",
                "integrate",
                "connect",
                "organize",
            ]
            if current_state == "initial" and any(
                keyword in message_lower for keyword in workflow_building_keywords
            ):
                logger.info(f"🚀 Detected WORKFLOW_BUILDING intent for: '{message}'")
                return {
                    "intent": "workflow_building",
                    "confidence": 0.95,
                    "entities": [],
                    "suggested_actions": ["Start questionnaire to build workflow"],
                }

            # TOOL INQUIRY INTENT - Check for questions about tools (when workflow_ready)
            # This allows users to ask "what are the tools for X", "show me tools for Y", etc.
            # This is different from explore_tools - it just returns info without adding
            tool_inquiry_keywords = [
                "what are the tools for",
                "what tools for",
                "which tools for",
                "what are the tools to",
                "what are the tools there to",
                "what are the tools that",
                "what are tools for",
                "show me tools for",
                "list tools for",
                "tell me tools for",
                "what tools can",
                "which tools can",
                "tools that can",
                "what tools do",
                "which tools do",
                "what tools there",
                "recommend tools for",
                "suggest tools for",
                "tools for",
                "show tools",
                "list tools",
                "tools to do",
                "tools there to do",
            ]

            # Only trigger in workflow_ready state for tool inquiries
            if current_state == "workflow_ready":
                # Check if message is asking about tools
                has_tool_inquiry = any(
                    keyword in message_lower for keyword in tool_inquiry_keywords
                )

                # Additional check: not an add/delete command
                action_keywords = [
                    "add",
                    "include",
                    "put",
                    "delete",
                    "remove",
                    "drop",
                    "create workflow",
                    "generate workflow",
                ]
                is_action = any(keyword in message_lower for keyword in action_keywords)

                logger.info(
                    f"🔍 Tool inquiry check: has_tool_inquiry={has_tool_inquiry}, is_action={is_action}, state={current_state}"
                )

                if has_tool_inquiry and not is_action:
                    logger.info(
                        f"🔍 Detected TOOL_INQUIRY intent for: '{message}' (state: {current_state})"
                    )
                    return {
                        "intent": "tool_inquiry",
                        "confidence": 0.95,
                        "entities": [],
                        "suggested_actions": [
                            "Search for tools and return list without adding to workflow"
                        ],
                    }

            # WORKFLOW QUESTION INTENT - Check for questions about the workflow (when workflow_ready)
            # This allows users to ask "what does this do?", "why is this connected?", etc.
            workflow_question_keywords = [
                "what is",
                "what does",
                "what are",
                "what's",
                "why is",
                "why does",
                "why are",
                "why",
                "how does",
                "how is",
                "how are",
                "how",
                "explain",
                "tell me about",
                "describe",
                "can you explain",
                "what do you mean",
                "node",
                "tool",
                "connection",
                "edge",
                "workflow",
                "this",
                "that",
            ]

            # Only trigger in workflow_ready state for question-like messages
            if current_state == "workflow_ready":
                # Check if message contains question keywords or ends with '?'
                is_question = message.strip().endswith("?")
                has_question_keyword = any(
                    keyword in message_lower for keyword in workflow_question_keywords
                )

                # Additional check: not an add/delete command
                action_keywords = [
                    "add",
                    "include",
                    "put",
                    "delete",
                    "remove",
                    "drop",
                    "create workflow",
                    "generate workflow",
                ]
                is_action = any(keyword in message_lower for keyword in action_keywords)

                if (is_question or has_question_keyword) and not is_action:
                    logger.info(f"❓ Detected WORKFLOW_QUESTION intent for: '{message}'")
                    return {
                        "intent": "workflow_question",
                        "confidence": 0.9,
                        "entities": [],
                        "suggested_actions": [
                            "Answer question about the workflow using workflow data"
                        ],
                    }

            # GREETING INTENT - Check SECOND (for initial state)
            # Only trigger if message is PURELY a greeting (no workflow keywords detected)
            greeting_keywords = [
                "hi",
                "hello",
                "hey",
                "greetings",
                "good morning",
                "good afternoon",
                "good evening",
                "howdy",
                "sup",
                "what's up",
                "yo",
            ]
            # Check if it's a greeting AND not a workflow building intent
            if current_state == "initial" and any(
                keyword in message_lower for keyword in greeting_keywords
            ):
                logger.info(f"👋 Detected GREETING intent for: '{message}'")
                return {
                    "intent": "greeting",
                    "confidence": 0.95,
                    "entities": [],
                    "suggested_actions": ["Respond with greeting and ask how to help"],
                }

            # WORKFLOW GENERATION TRIGGER - Check this THIRD (highest priority)
            # NOTE: This is now ONLY for workflow_ready state (regeneration)
            workflow_generation_keywords = [
                "create workflow",
                "generate workflow",
                "build workflow",
                "make workflow",
                "generate my workflow",
                "create my workflow",
                "build my workflow",
                "finish workflow",
                "complete workflow",
                "finalize workflow",
                "workflow for",
                "workflow on",  # "generate a workflow for X"
                "make me a workflow",
                "create me a workflow",
                "build me a workflow",
                "can you create",
                "can you generate",
                "can you build",  # Conversational variants
                "i need a workflow",
                "i want a workflow",
                "show me the workflow",
                "let's create",
                "let's generate",
                "ready to generate",
                "ready to create",
                "ready to build",
                "now generate",
                "now create",
                "now build",
                "go ahead and generate",
                "go ahead and create",
            ]
            if any(
                keyword in message_lower for keyword in workflow_generation_keywords
            ):
                logger.info(f"Detected GENERATE_WORKFLOW intent for: '{message}'")
                return {
                    "intent": "generate_workflow",
                    "confidence": 0.95,
                    "entities": [],
                    "suggested_actions": [
                        "Trigger workflow generation from conversation and nodes"
                    ],
                }

            # ADD TOOL INTENT - Check SECOND (explicit adding action)
            # Very specific keywords that indicate user wants to ADD something
            add_keywords = [
                "add",
                "include",
                "put",
                "insert",
                "append",
                "attach",
                "incorporate",
                "integrate",
            ]

            if any(keyword in message_lower for keyword in add_keywords):
                # Only allow if workflow is ready
                if current_state == "workflow_ready":
                    logger.info(
                        f"✅ Detected ADD_TOOL intent for: '{message}' (allowed)"
                    )
                    return {
                        "intent": "add_tool",
                        "confidence": 0.95,
                        "entities": self._extract_tool_names(message),
                        "suggested_actions": [
                            "Search for the tool and add it to workflow"
                        ],
                    }
                else:
                    logger.info(
                        f"🔒 ADD_TOOL intent blocked - not in workflow_ready state"
                    )
                    return {
                        "intent": "blocked_action",
                        "confidence": 0.9,
                        "entities": [],
                        "suggested_actions": [
                            "Inform user to complete questionnaire first"
                        ],
                    }

            # DELETE TOOL INTENT - Check THIRD (explicit deletion action)
            # Keywords that indicate user wants to DELETE/REMOVE something
            delete_keywords = [
                "delete",
                "remove",
                "drop",
                "eliminate",
                "discard",
                "take out",
                "get rid of",
                "uninstall",
                "erase",
            ]

            if any(keyword in message_lower for keyword in delete_keywords):
                # Only allow if workflow is ready
                if current_state == "workflow_ready":
                    logger.info(
                        f"✅ Detected DELETE_TOOL intent for: '{message}' (allowed)"
                    )
                    return {
                        "intent": "delete_tool",
                        "confidence": 0.95,
                        "entities": self._extract_tool_names(message),
                        "suggested_actions": ["Find and delete the tool from workflow"],
                    }
                else:
                    logger.info(
                        f"🔒 DELETE_TOOL intent blocked - not in workflow_ready state"
                    )
                    return {
                        "intent": "blocked_action",
                        "confidence": 0.9,
                        "entities": [],
                        "suggested_actions": [
                            "Inform user to complete questionnaire first"
                        ],
                    }

            # EXPLORATION INTENT - Check FOURTH (broader, generic queries)
            # Generic keywords that indicate user wants to EXPLORE or LEARN about tools
            exploration_keywords = [
                # Questions
                "what",
                "which",
                "how",
                "where",
                "when",
                "who",
                # Discovery
                "show",
                "find",
                "search",
                "look",
                "discover",
                "explore",
                # Requests
                "recommend",
                "suggest",
                "advise",
                "help",
                "need",
                "want",
                # Lists
                "list",
                "options",
                "choices",
                "alternatives",
                # General inquiries
                "tell me",
                "give me",
                "can you",
                "could you",
                # Tool-related
                "tools",
                "software",
                "solutions",
                "platforms",
                "apps",
            ]

            # Only allow exploration if workflow is ready
            if any(keyword in message_lower for keyword in exploration_keywords):
                if current_state == "workflow_ready":
                    # Double-check this isn't a tool inquiry that was missed
                    tool_inquiry_phrases = [
                        "what are the tools",
                        "what tools",
                        "which tools",
                        "show me tools",
                        "list tools",
                    ]
                    is_likely_inquiry = any(
                        phrase in message_lower for phrase in tool_inquiry_phrases
                    )

                    if is_likely_inquiry:
                        logger.info(
                            f"🔍 Exploration matched, but message looks like tool inquiry: '{message}'"
                        )
                        logger.info(
                            f"🔄 Routing to TOOL_INQUIRY instead of EXPLORE_TOOLS"
                        )
                        return {
                            "intent": "tool_inquiry",
                            "confidence": 0.9,
                            "entities": [],
                            "suggested_actions": [
                                "Search for tools and return list without adding to workflow"
                            ],
                        }

                    logger.info(
                        f"✅ Detected EXPLORE_TOOLS intent (allowed): '{message}'"
                    )
                    return {
                        "intent": "explore_tools",
                        "confidence": 0.85,
                        "entities": [],
                        "suggested_actions": [
                            "Search for relevant tools and show options"
                        ],
                    }
                else:
                    logger.info(
                        f"🔒 EXPLORE_TOOLS blocked - not in workflow_ready state"
                    )
                    return {
                        "intent": "blocked_action",
                        "confidence": 0.8,
                        "entities": [],
                        "suggested_actions": [
                            "Inform user to complete questionnaire first"
                        ],
                    }

            if any(
                keyword in message_lower
                for keyword in ["connect", "link", "join", "chain"]
            ):
                return {
                    "intent": "connect_tools",
                    "confidence": 0.8,
                    "entities": [],
                    "suggested_actions": ["Connect existing workflow nodes"],
                }

            if any(
                keyword in message_lower for keyword in ["workflow", "nodes", "tools"]
            ):
                return {
                    "intent": "workflow_discussion",
                    "confidence": 0.7,
                    "entities": [],
                    "suggested_actions": ["Discuss current workflow state"],
                }

            # Default to general chat
            return {
                "intent": "general_chat",
                "confidence": 0.6,
                "entities": [],
                "suggested_actions": [
                    "Continue conversation and provide helpful suggestions"
                ],
            }

        except Exception as e:
            logger.error(f"Error analyzing user intent: {e}")
            return {
                "intent": "general_chat",
                "confidence": 0.5,
                "entities": [],
                "suggested_actions": ["Continue conversation"],
            }

    def _extract_tool_names(self, message: str) -> List[str]:
        """Extract potential tool names from message."""
        # Simple extraction - look for capitalized words that might be tool names
        import re

        words = re.findall(r"\b[A-Z][a-zA-Z0-9-]+\b", message)
        return [word for word in words if len(word) > 2]

    async def generate_conversational_response(
        self, message: str, intent: str, context: Dict[str, Any]
    ) -> str:
        """Generate natural conversational responses."""
        try:
            response_prompt = f"""
            You are a helpful AI assistant helping users build workflows through conversation.

            User Message: "{message}"
            Intent: {intent}
            Context: {context}

            Generate a natural, conversational response that:
            1. Acknowledges what the user said
            2. Provides helpful information based on the intent
            3. Asks relevant follow-up questions
            4. Suggests next steps

            IMPORTANT:
            - Do NOT mention specific tool names (like Mailchimp, HubSpot, etc.) unless they are already in the workflow context
            - If the user asks about tools, suggest they use the workflow builder to search for tools
            - Keep responses conversational and general, not tool-specific

            Be conversational, friendly, and helpful. Don't be robotic.
            """

            return await self.llm.generate_response(response_prompt)

        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return "I understand. How can I help you with your workflow?"

    async def search_and_discuss_tools(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Search for tools and prepare discussion content."""
        try:
            search_result = await self.recommender.search_tools(
                query=query,
                max_results=max_results,
                include_pinecone=True,
                include_internet=True,
            )

            if search_result.get("status") == "error":
                return {
                    "tools": [],
                    "message": "Couldn't find any relevant tools for that query.",
                    "suggestions": [
                        "Try rephrasing your request",
                        "Be more specific about what you need",
                    ],
                }

            tools = search_result.get("tools", [])

            # Generate discussion points for each tool
            discussion_points = []
            for tool in tools[:3]:  # Focus on top 3 tools
                discussion_points.append(
                    {
                        "tool": tool,
                        "key_features": tool.get("Features", [])[:3],
                        "best_for": self._extract_use_cases(tool),
                        "why_relevant": f"Good for {query} because {tool.get('Description', '')[:100]}...",
                    }
                )

            return {
                "tools": tools,
                "discussion_points": discussion_points,
                "message": f"Found {len(tools)} tools that might help with your request.",
                "suggestions": [
                    "Explain any of these tools",
                    "Which tool interests you most?",
                    "Add any of these to your workflow",
                ],
            }

        except Exception as e:
            logger.error(f"Error searching and discussing tools: {e}")
            return {
                "tools": [],
                "message": "Encountered an error while searching for tools.",
                "suggestions": [
                    "Try again",
                    "Let me know if you need help with something else",
                ],
            }

    async def suggest_workflow_connections(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest connections between workflow nodes."""
        try:
            if len(nodes) < 2:
                return []

            # Analyze nodes to suggest logical connections
            connections = []
            for i, source_node in enumerate(nodes):
                for j, target_node in enumerate(nodes[i + 1 :], i + 1):
                    connection_strength = self._calculate_connection_strength(
                        source_node, target_node
                    )
                    if connection_strength > 0.3:  # Threshold for relevance
                        connections.append(
                            {
                                "source": source_node["id"],
                                "target": target_node["id"],
                                "reason": self._generate_connection_reason(
                                    source_node, target_node
                                ),
                                "strength": connection_strength,
                            }
                        )

            return connections

        except Exception as e:
            logger.error(f"Error suggesting workflow connections: {e}")
            return []

    async def generate_workflow_from_chat_history(
        self, chat_history: List[Dict], original_query: str
    ) -> Dict[str, Any]:
        """Generate workflow from entire chat history."""
        try:
            # Analyze chat history to extract tools and intent
            chat_analysis = self._analyze_chat_history(chat_history)

            # Generate workflow using the recommender
            workflow = await self.recommender.generate_workflow(
                original_query, chat_analysis["extracted_tools"]
            )

            return workflow

        except Exception as e:
            logger.error(f"Error generating workflow from chat history: {e}")
            return None

    def _prepare_chat_context(self, chat_history: List[Dict]) -> str:
        """Prepare chat context for AI analysis."""
        if not chat_history:
            return "No previous conversation"

        # Get last 5 messages for context
        recent_messages = chat_history[-5:]
        context = "Recent conversation:\n"
        for msg in recent_messages:
            context += f"User: {msg.get('user', '')}\n"
            context += f"AI: {msg.get('ai', '')}\n\n"

        return context

    def _extract_use_cases(self, tool: Dict[str, Any]) -> List[str]:
        """Extract use cases from tool data."""
        features = tool.get("Features", [])

        # Simple extraction of use cases
        use_cases = []
        if "automation" in str(features).lower():
            use_cases.append("automation")
        if "crm" in str(features).lower():
            use_cases.append("customer management")
        if "email" in str(features).lower():
            use_cases.append("email marketing")

        return use_cases[:3]  # Limit to 3 use cases

    def _calculate_connection_strength(
        self, source_node: Dict[str, Any], target_node: Dict[str, Any]
    ) -> float:
        """Calculate how well two nodes should be connected."""
        try:
            source_features = source_node.get("data", {}).get("features", [])
            target_features = target_node.get("data", {}).get("features", [])

            # Simple similarity calculation
            source_text = " ".join(source_features).lower()
            target_text = " ".join(target_features).lower()

            # Check for common keywords
            common_keywords = ["automation", "crm", "email", "marketing", "sales"]
            matches = sum(
                1
                for keyword in common_keywords
                if keyword in source_text and keyword in target_text
            )

            return matches / len(common_keywords)

        except Exception:
            return 0.0

    def _generate_connection_reason(
        self, source_node: Dict[str, Any], target_node: Dict[str, Any]
    ) -> str:
        """Generate human-readable reason for connection."""
        source_name = source_node.get("data", {}).get("label", "Tool")
        target_name = target_node.get("data", {}).get("label", "Tool")

        return f"{source_name} can feed data into {target_name} for a complete workflow"

    def _analyze_chat_history(self, chat_history: List[Dict]) -> Dict[str, Any]:
        """Analyze chat history to extract tools and intent."""
        extracted_tools = []
        user_intent = "workflow_building"

        for message in chat_history:
            # Extract tools mentioned in AI responses
            ai_response = message.get("ai", "")
            if "tool" in ai_response.lower():
                # Simple extraction - in real implementation, use NLP
                pass

        return {
            "extracted_tools": extracted_tools,
            "user_intent": user_intent,
            "conversation_length": len(chat_history),
        }

    async def format_tools_response(
        self, tools: List[Dict[str, Any]], user_message: str
    ) -> Dict[str, str]:
        """
        Generate dynamic response using LLM based on tools found.
        NO HARDCODED STRINGS - everything is AI-generated!

        Args:
            tools: List of tools found
            user_message: Original user message for context

        Returns:
            Dict with AI-generated message and suggestions
        """
        try:
            if not tools:
                # Generate AI response for no tools found
                prompt = f"""
                User asked: "{user_message}"

                No tools were found matching their request.

                Generate a casual response that:
                1. Tells them no tools were found
                2. Suggests they be more specific

                Be conversational like texting. NO emojis, NO exclamation marks.
                Keep it short (1-2 sentences).
                """

                message = await self.llm.generate_response(prompt)

                return {
                    "message": message.strip(),
                    "suggestions": [
                        "Try a more specific request",
                        "Describe your goal in detail",
                        "Search for different tools",
                    ],
                }

            # Build tool information for AI to use
            tools_info = []
            for i, tool in enumerate(tools[:3], 1):
                tool_name = (
                    tool.get("Title")
                    or tool.get("title")
                    or tool.get("name")
                    or "Unknown Tool"
                )
                tool_desc = (
                    tool.get("Description")
                    or tool.get("description")
                    or "No description available"
                )
                tool_website = tool.get("Website") or tool.get("website") or ""

                tools_info.append(
                    {
                        "number": i,
                        "name": tool_name,
                        "description": tool_desc[:200],  # Limit for context
                        "website": tool_website,
                    }
                )

            # Generate AI response with tool details
            prompt = f"""
            User asked: "{user_message}"

            Found {len(tools)} tools. Here are the top {len(tools_info)}:

            {json.dumps(tools_info, indent=2)}

            Generate a casual response that:
            1. Lists the tools with their names and brief descriptions
            2. Mentions if they want to add any

            Format as:
            [Simple intro about finding tools]

            **1. [Tool Name]**
               [Brief description]
               [website if available]

            **2. [Tool Name]**
            ...

            [Simple closing about adding to workflow]

            Be conversational like texting. NO emojis, NO exclamation marks.
            Keep it informative but casual.
            """

            message = await self.llm.generate_response(prompt)

            # Generate suggestions dynamically
            first_tool_name = (
                tools[0].get("Title")
                or tools[0].get("title")
                or tools[0].get("name")
                or "this tool"
            )
            suggestions = [
                f"Add {first_tool_name} to workflow",
                "Show me more options",
                "Explain how these work together",
            ]

            return {"message": message.strip(), "suggestions": suggestions}

        except Exception as e:
            logger.error(f"Error generating AI response for tools: {e}")
            # Simple fallback that just lists tool names
            if tools:
                tool_names = [
                    t.get("Title") or t.get("title") or t.get("name") or "Tool"
                    for t in tools[:3]
                ]
                return {
                    "message": f"Found these tools: {', '.join(tool_names)}. Want to add any?",
                    "suggestions": ["Add to workflow", "Show details", "Search again"],
                }
            else:
                return {
                    "message": "Couldn't find tools for your request. Try being more specific.",
                    "suggestions": [
                        "Try again",
                        "Describe your needs",
                        "Search differently",
                    ],
                }

    async def answer_workflow_question(
        self,
        user_question: str,
        workflow_nodes: List[Dict],
        workflow_edges: List[Dict],
        original_query: str,
    ) -> str:
        """
        Answer user questions about the workflow using LLM.

        Args:
            user_question: The question the user is asking
            workflow_nodes: List of workflow nodes with all their data
            workflow_edges: List of workflow edges/connections
            original_query: Original query that created the workflow

        Returns:
            LLM-generated answer to the question
        """
        try:
            # Format workflow data for LLM
            nodes_summary = []
            for i, node in enumerate(workflow_nodes, 1):
                node_data = node.get("data", {})
                nodes_summary.append(
                    {
                        "number": i,
                        "id": node.get("id"),
                        "name": node_data.get("label", "Unknown Tool"),
                        "description": node_data.get("description", ""),
                        "features": node_data.get("features", []),
                        "tags": node_data.get("tags", []),
                    }
                )

            # Format edges/connections
            connections_summary = []
            for edge in workflow_edges:
                # Find source and target node names
                source_node = next(
                    (n for n in workflow_nodes if n.get("id") == edge.get("source")),
                    None,
                )
                target_node = next(
                    (n for n in workflow_nodes if n.get("id") == edge.get("target")),
                    None,
                )

                if source_node and target_node:
                    connections_summary.append(
                        {
                            "from": source_node.get("data", {}).get("label", "Unknown"),
                            "to": target_node.get("data", {}).get("label", "Unknown"),
                            "type": edge.get("type", "default"),
                        }
                    )

            # Create comprehensive prompt for LLM
            prompt = f"""
            You are helping a user understand their workflow. Answer their question clearly and conversationally.

            **User's Question:** "{user_question}"

            **Original Goal:** {original_query}

            **Workflow Tools ({len(nodes_summary)} total):**
            {self._format_json_for_prompt(nodes_summary)}

            **Connections ({len(connections_summary)} total):**
            {self._format_json_for_prompt(connections_summary)}

            Instructions:
            1. Answer the user's specific question based on the workflow data
            2. Be conversational and friendly
            3. Reference specific tool names and features when relevant
            4. If they ask about "this" or "that", infer from context what they mean
            5. If they ask about a specific tool or node, provide detailed info about it
            6. If they ask about connections, explain how tools work together
            7. If they ask "why", explain the reasoning and benefits
            8. If they ask "what", describe the functionality
            9. If they ask "how", explain the process and data flow
            10. Keep your answer focused on their question - don't ramble

            Be helpful and specific. Use the workflow data to give accurate answers.
            """

            response = await self.llm.generate_response(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Error answering workflow question: {e}")
            return (
                "I understand you have a question about the workflow. "
                "Let me know what specifically you'd like to know - "
                "I can explain what each tool does, why tools are connected, "
                "or how the workflow helps achieve your goals."
            )

    def _format_json_for_prompt(self, data) -> str:
        """Format JSON data nicely for LLM prompt."""
        try:
            import json

            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

    async def format_tool_inquiry_response(
        self, tools: List[Dict[str, Any]], user_message: str
    ) -> Dict[str, str]:
        """
        Generate response for tool inquiry - shows tools without adding them.
        This is different from format_tools_response which suggests adding.

        Args:
            tools: List of tools found
            user_message: Original user message for context

        Returns:
            Dict with AI-generated message and suggestions
        """
        try:
            if not tools:
                # Generate AI response for no tools found
                prompt = f"""
                User asked: "{user_message}"

                No tools were found matching their inquiry.

                Generate a brief response that:
                1. Tells them no tools were found
                2. Suggests they try a different search

                Be conversational like texting. NO emojis, NO exclamation marks.
                Keep it short (1-2 sentences).
                """

                message = await self.llm.generate_response(prompt)

                return {
                    "message": message.strip(),
                    "suggestions": [
                        "Try a different search",
                        "Be more specific",
                        "Ask about other tools",
                    ],
                }

            # Build tool information for AI to use
            tools_info = []
            for i, tool in enumerate(tools[:5], 1):  # Show top 5
                tool_name = (
                    tool.get("Title")
                    or tool.get("title")
                    or tool.get("name")
                    or "Unknown Tool"
                )
                tool_desc = (
                    tool.get("Description")
                    or tool.get("description")
                    or "No description available"
                )
                tool_website = tool.get("Website") or tool.get("website") or ""
                tool_features = tool.get("Features") or tool.get("features") or []

                tools_info.append(
                    {
                        "number": i,
                        "name": tool_name,
                        "description": tool_desc[:200],  # Limit for context
                        "website": tool_website,
                        "features": tool_features[:3]
                        if isinstance(tool_features, list)
                        else [],
                    }
                )

            # Generate AI response with tool details
            prompt = f"""
            User inquired: "{user_message}"

            Found {len(tools)} relevant tools. Here are the top {len(tools_info)}:

            {json.dumps(tools_info, indent=2)}

            Generate an informative response that:
            1. Lists the tools with their names and key features
            2. Describes what each tool does briefly
            3. Mentions if they're interested in adding any

            Format as a clean list:
            Here are the tools for [their need]:

            **1. [Tool Name]**
               [Brief description]
               Key features: [feature 1], [feature 2]
               Website: [url]

            **2. [Tool Name]**
            ...

            [Simple closing about adding to workflow if interested]

            Be informative and professional. NO emojis, NO exclamation marks.
            """

            message = await self.llm.generate_response(prompt)

            # Generate suggestions dynamically
            first_tool_name = (
                tools[0].get("Title")
                or tools[0].get("title")
                or tools[0].get("name")
                or "this tool"
            )
            suggestions = [
                f"Add {first_tool_name}",
                "Show me more tools",
                "Compare these tools",
                "Tell me about another category",
            ]

            return {
                "message": message.strip(),
                "suggestions": suggestions,
                "tools": tools[:5],  # Return top 5 tools
            }

        except Exception as e:
            logger.error(f"Error generating tool inquiry response: {e}")
            # Simple fallback that just lists tool names
            if tools:
                tool_names = [
                    t.get("Title") or t.get("title") or t.get("name") or "Tool"
                    for t in tools[:5]
                ]
                return {
                    "message": f"Here are the tools I found: {', '.join(tool_names)}. Let me know if you'd like details on any of them.",
                    "suggestions": [
                        "Tell me more",
                        "Add a tool",
                        "Search for something else",
                    ],
                    "tools": tools[:5],
                }
            else:
                return {
                    "message": "I couldn't find tools matching your inquiry. Try being more specific about what you need.",
                    "suggestions": [
                        "Try again",
                        "Ask differently",
                        "Search for other tools",
                    ],
                    "tools": [],
                }

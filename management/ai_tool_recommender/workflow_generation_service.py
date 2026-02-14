"""Service for generating complete workflows from conversational context."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async

from .ai_agents.core.llm import get_shared_llm
from .ai_agents.tools.ai_tool_recommender import AIToolRecommender

logger = logging.getLogger(__name__)


class WorkflowGenerationService:
    """Service to generate workflows from conversation context and nodes."""

    def __init__(self):
        """Initialize the workflow generation service."""
        self.llm = get_shared_llm()
        self.recommender = AIToolRecommender()

    def _create_tool_node_from_dict(
        self, tool: Dict[str, Any], sequence: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create a standardized tool node from tool dictionary.

        Args:
            tool: Tool dictionary with Title, Description, Features, etc.
            sequence: Sequence number for this node (1-based)

        Returns:
            Node dictionary with id, type, data, and position
        """
        node_id = str(uuid.uuid4())

        # Extract and clean description (remove markdown formatting)
        raw_description = tool.get("Description", tool.get("description", ""))
        clean_description = self._clean_description(raw_description)

        # Extract and normalize features (ensure it's an array)
        features = tool.get("Features", tool.get("features", []))
        if isinstance(features, str):
            # Split comma-separated string into array
            features = [f.strip() for f in features.split(",") if f.strip()]

        # Extract and normalize tags (ensure it's an array)
        tags = tool.get("Tags (Keywords)", tool.get("tags", []))
        if isinstance(tags, str):
            # Split comma-separated string into array
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        # Validate tool name first
        tool_name = tool.get("Title", tool.get("title", ""))

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
            logger.warning(
                f"❌ Rejecting workflow node with garbage name: '{tool_name}'"
            )
            return None  # Return None to indicate this tool should be skipped

        # Generate recommendation reason based on tool data and sequence
        recommendation_reason = self._generate_node_recommendation_reason(
            tool_name, clean_description, features, sequence
        )

        # Extract and validate official website URL (prioritize "Website" field)
        website = (tool.get("Website", "") or tool.get("website", "") or "").strip()

        # Validate website URL - must be a valid URL or empty string
        if website and not (
            website.startswith("http://") or website.startswith("https://")
        ):
            # If website is not a valid URL (e.g., contains tool name), clear it
            logger.warning(
                f"Invalid website field for tool '{tool_name}': '{website}' - clearing it"
            )
            website = ""
        elif website:
            # Log when we successfully use official website
            logger.debug(f"✅ Using official website for '{tool_name}': {website}")

        return {
            "id": node_id,
            "type": "tool",
            "data": {
                "label": tool_name,
                "description": clean_description,
                "features": features,
                "tags": tags,
                "recommendation_reason": recommendation_reason,
                "website": website,
                "twitter": tool.get("Twitter", tool.get("twitter", "")),
                "facebook": tool.get("Facebook", tool.get("facebook", "")),
                "linkedin": tool.get("LinkedIn", tool.get("linkedin", "")),
                "instagram": tool.get("Instagram", tool.get("instagram", "")),
                "source": tool.get("Source", "Pinecone Vector Database"),
                "auto_generated": True,
                "sequence": sequence,
            },
            "position": {
                "x": 100 + ((sequence - 1) % 3) * 250,
                "y": 100 + ((sequence - 1) // 3) * 150,
            },
        }

    def _generate_node_recommendation_reason(
        self, tool_name: str, description: str, features: List[str], sequence: int
    ) -> str:
        """
        Generate a workflow-specific recommendation reason for a tool node.

        Args:
            tool_name: Name of the tool
            description: Tool description
            features: List of tool features
            sequence: Position in workflow

        Returns:
            Workflow-specific recommendation reason string
        """
        # Try to extract key benefits from description
        desc_lower = description.lower() if description else ""

        # Keywords that indicate workflow value and contribution
        value_keywords = {
            "automate": "automates key processes",
            "manage": "manages critical workflows",
            "track": "tracks important metrics",
            "analyze": "analyzes workflow data",
            "create": "creates essential content",
            "integrate": "integrates with other tools",
            "optimize": "optimizes workflow efficiency",
            "collaborate": "enables team collaboration",
            "monitor": "monitors workflow progress",
            "schedule": "schedules automated tasks",
        }

        # Find matching value propositions
        found_values = []
        for keyword, value in value_keywords.items():
            if keyword in desc_lower:
                found_values.append(value)

        # Generate workflow-specific reason based on what we found
        if found_values:
            primary_value = found_values[0]
            return f"Chosen for your workflow because it {primary_value}, helping you achieve your automation goals more efficiently"
        elif features and len(features) > 0:
            # Use first feature if available
            first_feature = features[0] if isinstance(features, list) else str(features)
            return f"Selected for your workflow because its {first_feature} capabilities directly support your specific automation needs"
        else:
            # Generic but workflow-focused
            return f"Included in your workflow because {tool_name} provides essential functionality that contributes to achieving your automation objectives"

    def _clean_description(self, description: str) -> str:
        """
        Clean markdown formatting from description.

        Args:
            description: Raw description potentially with markdown

        Returns:
            Clean description without markdown
        """
        if not description:
            return ""

        import re

        # Remove markdown formatting like **Title:** **Website:** **Features:**
        # Pattern matches lines starting with **text:**
        description = re.sub(r"\*\*[^*]+:\*\*\s*", "", description)

        # Remove markdown links [text](url)
        description = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", description)

        # Remove remaining ** bold markers
        description = re.sub(r"\*\*([^*]+)\*\*", r"\1", description)

        # Remove multiple newlines and extra whitespace
        description = re.sub(r"\n\s*\n", "\n", description)
        description = description.strip()

        return description

    def _create_node_identifier(self, tool_name: str, website: str) -> str:
        """
        Create a unique identifier for workflow nodes to prevent duplicates.
        Uses multiple strategies: website domain, normalized name, etc.

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
                normalized_name = (
                    tool_name.lower()
                    .strip()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                    .replace(".", "")
                    .replace("|", "")
                    .replace(":", "")
                    .replace("/", "")
                    # Remove common suffixes that don't affect uniqueness
                    .replace("ai", "")
                    .replace("app", "")
                    .replace("tool", "")
                    .replace("software", "")
                )
                # Only use if we have a meaningful name (at least 3 chars)
                if len(normalized_name) >= 3:
                    return f"name:{normalized_name}"

            # Strategy 3: Fallback to original name (normalized)
            return f"original:{tool_name.lower().strip()}" if tool_name else "unknown"

        except Exception as e:
            logger.error(f"Error creating node identifier: {e}")
            return f"fallback:{tool_name.lower().strip()}" if tool_name else "error"

    async def generate_workflow_from_conversation(
        self,
        conversation_session,
        request_user,
        workflow_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete workflow from conversation context and nodes.

        Args:
            conversation_session: ConversationSession model instance
            request_user: User making the request
            workflow_id: Optional existing workflow ID to update

        Returns:
            Dict containing workflow, metadata, and response message
        """
        try:
            logger.info(
                f"🚀 Starting workflow generation for session {conversation_session.session_id}"
            )

            # Check if we have nodes or need to generate from chat
            if (
                not conversation_session.workflow_nodes
                and conversation_session.chat_history
            ):
                logger.info("📝 No nodes found, generating workflow from chat history")
                return await self._generate_workflow_from_chat_only(
                    conversation_session, request_user, workflow_id
                )

            # Step 1: Analyze conversation context
            context_analysis = await self._analyze_conversation_context(
                conversation_session.chat_history,
                conversation_session.original_query,
                conversation_session.workflow_nodes,
            )

            logger.info(
                f"📊 Context analysis: {len(conversation_session.workflow_nodes)} nodes, "
                f"{len(conversation_session.chat_history)} chat messages"
            )

            # Step 2: Filter and validate nodes
            validated_nodes, rejected_nodes = await self._validate_and_filter_nodes(
                conversation_session.workflow_nodes, context_analysis
            )

            logger.info(
                f"✅ Validated {len(validated_nodes)} nodes, ❌ Rejected {len(rejected_nodes)} nodes"
            )

            # Step 3: Generate edges using intelligent workflow generation
            from .ai_agents.tools.ai_tool_recommender import AIToolRecommender

            recommender = AIToolRecommender()

            # Get refined query for better context if available
            refined_query = None
            try:
                from .models import RefinedQuery

                refined_query_obj = await sync_to_async(
                    lambda: RefinedQuery.objects.filter(workflow_id=workflow_id).first()
                )()
                if refined_query_obj:
                    refined_query = refined_query_obj.refined_query
                    logger.info(
                        f"📋 Using refined query for edge generation: {refined_query[:100]}..."
                    )
            except Exception as e:
                logger.warning(f"Could not fetch refined query: {e}")

            # Convert nodes to tools format that generate_workflow expects
            tools_for_workflow = []
            for node in validated_nodes:
                node_data = node.get("data", {})
                tools_for_workflow.append(
                    {
                        "Title": node_data.get("label", "Tool"),
                        "Description": node_data.get("description", ""),
                        "Features": node_data.get("features", []),
                        "Tags (Keywords)": node_data.get("tags", []),
                        "Website": node_data.get("website", ""),
                        "Source": node_data.get("source", "Conversation"),
                    }
                )

            # Use the search API's workflow generation to create meaningful edges
            query = (
                refined_query
                or conversation_session.original_query
                or "Connect these tools in a meaningful workflow"
            )
            logger.info(f"🔗 Generating intelligent edges for query: {query[:100]}...")

            generated_workflow = await recommender.generate_workflow(
                query, tools_for_workflow, refined_query=refined_query
            )

            # Extract edges from the generated workflow and map node IDs
            generated_edges = (
                generated_workflow.get("edges", []) if generated_workflow else []
            )
            generated_nodes = (
                generated_workflow.get("nodes", []) if generated_workflow else []
            )

            logger.info(
                f"📊 Generated workflow has {len(generated_nodes)} nodes and {len(generated_edges)} edges"
            )
            logger.info(f"📊 We have {len(validated_nodes)} validated nodes")

            # IMPROVED: Map edges by tool name/label instead of index for better reliability
            edges = []
            if generated_edges and generated_nodes:
                # Create mapping from generated node labels to our actual node IDs
                label_to_node_id = {}
                for node in validated_nodes:
                    label = node.get("data", {}).get("label", "").strip().lower()
                    if label:
                        label_to_node_id[label] = node.get("id")

                # Create mapping from generated nodes (by label) to their IDs
                generated_label_to_id = {}
                for gen_node in generated_nodes:
                    gen_label = (
                        gen_node.get("data", {}).get("label", "").strip().lower()
                    )
                    if gen_label:
                        generated_label_to_id[gen_label] = gen_node.get("id")

                # Map edges by matching tool names
                mapped_count = 0
                for edge in generated_edges:
                    source_gen_id = edge.get("source")
                    target_gen_id = edge.get("target")

                    # Find the generated nodes by their IDs
                    source_gen_node = next(
                        (n for n in generated_nodes if n.get("id") == source_gen_id),
                        None,
                    )
                    target_gen_node = next(
                        (n for n in generated_nodes if n.get("id") == target_gen_id),
                        None,
                    )

                    if source_gen_node and target_gen_node:
                        source_label = (
                            source_gen_node.get("data", {})
                            .get("label", "")
                            .strip()
                            .lower()
                        )
                        target_label = (
                            target_gen_node.get("data", {})
                            .get("label", "")
                            .strip()
                            .lower()
                        )

                        # Find matching actual node IDs
                        actual_source_id = label_to_node_id.get(source_label)
                        actual_target_id = label_to_node_id.get(target_label)

                        if (
                            actual_source_id
                            and actual_target_id
                            and actual_source_id != actual_target_id
                        ):
                            # Valid edge with both nodes found and no self-loops
                            edges.append(
                                {
                                    "id": edge.get("id", str(uuid.uuid4())),
                                    "source": actual_source_id,
                                    "target": actual_target_id,
                                    "type": edge.get("type", "default"),
                                    "animated": edge.get("animated", False),
                                }
                            )
                            mapped_count += 1
                            logger.debug(
                                f"✅ Mapped edge: {source_label[:30]} → {target_label[:30]}"
                            )
                        else:
                            logger.warning(
                                f"⚠️ Could not map edge: {source_label[:30]} → {target_label[:30]}"
                            )

                logger.info(
                    f"✅ Successfully mapped {mapped_count}/{len(generated_edges)} intelligent edges"
                )

            # IMPROVED: Only use fallback if we have very few edges, and enhance it with context
            min_edges_needed = max(
                1, len(validated_nodes) - 1
            )  # At least n-1 for connectivity

            if len(edges) < min_edges_needed and len(validated_nodes) > 1:
                logger.warning(
                    f"⚠️ Insufficient intelligent edges ({len(edges)}/{min_edges_needed}), "
                    f"generating context-aware connections..."
                )
                # Use intelligent edge generation with context instead of simple similarity
                context_aware_edges = (
                    await self._generate_intelligent_edges_with_context(
                        validated_nodes, context_analysis, query
                    )
                )

                # Merge intelligent edges with any we already have
                existing_edge_pairs = {(e["source"], e["target"]) for e in edges}
                for edge in context_aware_edges:
                    edge_pair = (edge["source"], edge["target"])
                    if edge_pair not in existing_edge_pairs:
                        edges.append(edge)
                        existing_edge_pairs.add(edge_pair)

                logger.info(
                    f"✅ Generated {len(edges)} total edges ({len(context_aware_edges)} context-aware + {mapped_count} from LLM)"
                )

            logger.info(
                f"🔗 Final edge count: {len(edges)} edges for {len(validated_nodes)} nodes"
            )

            # Step 4: Optimize node positions for visualization
            positioned_nodes = await self._optimize_node_positions(validated_nodes)

            # Step 5: Create complete workflow structure
            workflow_data = {
                "nodes": positioned_nodes,
                "edges": edges,
                "metadata": {
                    "generated_from": "conversation",
                    "original_query": conversation_session.original_query,
                    "context_summary": context_analysis.get("summary", ""),
                    "rejected_nodes": rejected_nodes,
                    "generation_method": "ai_analysis",
                },
            }

            # Step 6: Save workflow to database
            saved_workflow_id = await self._save_workflow_to_database(
                workflow_data, conversation_session, request_user, workflow_id
            )

            # Step 7: Update conversation session with final workflow
            logger.info(f"💾 Updating conversation session with workflow data")
            conversation_session.workflow_edges = edges
            conversation_session.workflow_nodes = positioned_nodes
            await sync_to_async(conversation_session.save)()
            logger.info(f"✅ Conversation session updated successfully")

            # Step 8: Generate response message
            response_message = await self._generate_response_message(
                validated_nodes, rejected_nodes, edges, context_analysis
            )

            logger.info(f"✅ Successfully generated workflow {saved_workflow_id}")

            result = {
                "status": "success",
                "workflow": workflow_data,
                "workflow_id": str(saved_workflow_id),
                "message": response_message,
                "metadata": {
                    "total_nodes": len(positioned_nodes),
                    "total_edges": len(edges),
                    "rejected_nodes": len(rejected_nodes),
                    "rejected_node_details": rejected_nodes,
                },
            }

            logger.info(f"📦 Returning workflow result with ID: {result['workflow_id']}")
            return result

        except Exception as e:
            logger.error(f"❌ Error generating workflow: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to generate workflow: {str(e)}",
                "workflow": None,
                "workflow_id": None,
            }

    async def _analyze_conversation_context(
        self, chat_history: List[Dict], original_query: str, workflow_nodes: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze conversation context to understand workflow intent."""
        try:
            # Prepare context for LLM analysis
            context_prompt = f"""
            Analyze this conversation and workflow building context:

            Original Query: {original_query}

            Chat History ({len(chat_history)} messages):
            {self._format_chat_history(chat_history[-10:])}  # Last 10 messages

            Current Nodes ({len(workflow_nodes)}):
            {self._format_nodes_for_analysis(workflow_nodes)}

            Provide a JSON analysis with:
            1. workflow_intent: What is the user trying to build?
            2. workflow_type: Type of workflow (sequential, parallel, conditional, etc.)
            3. key_requirements: Main requirements mentioned in conversation
            4. node_relationships: How should nodes be connected?
            5. summary: Brief summary of the workflow purpose

            Return ONLY valid JSON.
            """

            # Use LLM to analyze context
            response = await self.llm.generate_response(context_prompt)
            analysis_result = await self.llm.parse_json_response(response)

            return analysis_result or {
                "workflow_intent": "General workflow",
                "workflow_type": "sequential",
                "key_requirements": [],
                "node_relationships": [],
                "summary": "Automated workflow based on user conversation",
            }

        except Exception as e:
            logger.error(f"Error analyzing conversation context: {e}")
            return {
                "workflow_intent": "General workflow",
                "workflow_type": "sequential",
                "key_requirements": [],
                "node_relationships": [],
                "summary": "Automated workflow",
            }

    async def _validate_and_filter_nodes(
        self, workflow_nodes: List[Dict], context_analysis: Dict[str, Any]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Validate and filter nodes, removing duplicates while keeping all unique nodes.

        Returns:
            Tuple of (validated_nodes, rejected_nodes)
        """
        try:
            if not workflow_nodes:
                return [], []

            validated_nodes = []
            rejected_nodes = []
            seen_identifiers = set()

            for node in workflow_nodes:
                node_data = node.get("data", {})
                tool_name = node_data.get("label", "").strip()
                website = node_data.get("website", "").strip()

                # Skip nodes with invalid names
                garbage_names = [
                    "tool",
                    "unknown tool",
                    "untitled",
                    "unknown",
                    "",
                    "ai tool",
                    "software",
                ]
                if not tool_name or tool_name.lower() in garbage_names:
                    logger.warning(f"❌ Rejecting node with invalid name: '{tool_name}'")
                    rejected_nodes.append(
                        {"node": node, "reason": f"Invalid tool name: '{tool_name}'"}
                    )
                    continue

                # Create comprehensive identifier for duplicate detection
                identifier = self._create_node_identifier(tool_name, website)

                # Check if this is a duplicate
                if identifier in seen_identifiers:
                    logger.warning(
                        f"🔄 Skipping duplicate node: '{tool_name}' (identifier: {identifier})"
                    )
                    rejected_nodes.append(
                        {
                            "node": node,
                            "reason": f"Duplicate of existing tool: '{tool_name}'",
                        }
                    )
                    continue

                # Add to seen set and validated list
                seen_identifiers.add(identifier)
                validated_nodes.append(node)
                logger.debug(
                    f"✅ Validated node: '{tool_name}' (identifier: {identifier})"
                )

            logger.info(
                f"✅ Validated {len(validated_nodes)} unique nodes, "
                f"❌ Rejected {len(rejected_nodes)} duplicates/invalid nodes"
            )

            return validated_nodes, rejected_nodes

        except Exception as e:
            logger.error(f"Error in node validation: {e}", exc_info=True)
            # On error, return all nodes but log the issue
            return workflow_nodes, []

    async def _generate_workflow_edges(
        self, validated_nodes: List[Dict], context_analysis: Dict[str, Any]
    ) -> List[Dict]:
        """Generate edges/connections between workflow nodes."""
        try:
            if len(validated_nodes) == 0:
                return []

            if len(validated_nodes) == 1:
                # Single node: no edges needed
                return []

            workflow_type = context_analysis.get("workflow_type", "sequential")

            if workflow_type == "sequential":
                # Use helper method for similarity-based workflow edges
                return await self._generate_similarity_based_edges(validated_nodes)

            else:
                # For other workflow types, use AI to determine connections
                edges_prompt = f"""
                Workflow Type: {workflow_type}

                Nodes:
                {self._format_nodes_for_analysis(validated_nodes)}

                Generate appropriate edges/connections for this workflow.
                Return JSON with:
                {{
                    "edges": [
                        {{
                            "source": "source_node_id",
                            "target": "target_node_id",
                            "type": "default",
                            "reason": "why these nodes connect"
                        }}
                    ]
                }}

                Return ONLY valid JSON.
                """

                response = await self.llm.generate_response(edges_prompt)
                edges_result = await self.llm.parse_json_response(response)

                if not edges_result or "edges" not in edges_result:
                    # Fallback to similarity-based edges
                    return await self._generate_similarity_based_edges(validated_nodes)

                # Format edges with UUIDs - NO self-loops!
                edges = []

                # AI-generated connection edges
                for edge_data in edges_result["edges"]:
                    # Skip self-loops (source == target)
                    if edge_data["source"] == edge_data["target"]:
                        continue

                    edge = {
                        "id": str(uuid.uuid4()),  # Generate UUID for edge
                        "source": edge_data["source"],
                        "target": edge_data["target"],
                        "type": edge_data.get("type", "default"),
                        "animated": False,
                    }
                    edges.append(edge)

                # That's it! Just use the AI-generated edges
                # NO self-loops added

                return edges

        except Exception as e:
            logger.error(f"Error generating edges: {e}")
            # Fallback to similarity-based edges
            return await self._generate_similarity_based_edges(validated_nodes)

    async def regenerate_edges_intelligently(
        self,
        nodes: List[Dict],
        original_query: str = "",
        workflow_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Regenerate edges intelligently based on tool functionality and relationships.
        This method is called when nodes are added or removed to reconnect the workflow.

        Args:
            nodes: List of workflow nodes
            original_query: Optional query context for better edge generation
            workflow_id: Optional workflow ID to fetch refined query

        Returns:
            List of intelligently generated edges
        """
        try:
            if len(nodes) == 0:
                return []

            if len(nodes) == 1:
                # Single node: no edges needed
                logger.info("Only 1 node, no edges needed")
                return []

            logger.info(f"🔄 Regenerating edges intelligently for {len(nodes)} nodes")

            # Try to get refined query for better context
            refined_query = None
            if workflow_id:
                try:
                    from .models import RefinedQuery

                    refined_query_obj = await sync_to_async(
                        lambda: RefinedQuery.objects.filter(
                            workflow_id=workflow_id
                        ).first()
                    )()
                    if refined_query_obj:
                        refined_query = refined_query_obj.refined_query
                        logger.info(
                            f"📋 Using refined query for edge regeneration: {refined_query[:100]}..."
                        )
                except Exception as e:
                    logger.warning(f"Could not fetch refined query: {e}")

            # Convert nodes to tools format for workflow generation API
            tools_for_workflow = []
            for node in nodes:
                node_data = node.get("data", {})
                tools_for_workflow.append(
                    {
                        "Title": node_data.get("label", "Tool"),
                        "Description": node_data.get("description", ""),
                        "Features": node_data.get("features", []),
                        "Tags (Keywords)": node_data.get("tags", []),
                        "Website": node_data.get("website", ""),
                        "Source": node_data.get("source", "Conversation"),
                    }
                )

            # Use the search API's workflow generation to create meaningful edges
            query = (
                refined_query
                or original_query
                or "Connect these tools in an intelligent workflow based on their functionality"
            )
            logger.info(f"🔗 Regenerating edges with query: {query[:100]}...")

            generated_workflow = await self.recommender.generate_workflow(
                query, tools_for_workflow, refined_query=refined_query
            )

            # Extract edges from the generated workflow
            if generated_workflow and generated_workflow.get("edges"):
                generated_edges = generated_workflow.get("edges", [])
                generated_nodes = generated_workflow.get("nodes", [])

                logger.info(
                    f"📊 API generated {len(generated_edges)} edges for {len(generated_nodes)} nodes"
                )

                # Create a mapping from generated node labels to our actual node IDs
                # Match by tool name/label for more reliable mapping
                edges = []

                # Build label-to-node mapping for our actual nodes
                label_to_node = {}
                for node in nodes:
                    label = node.get("data", {}).get("label", "").lower().strip()
                    if label:
                        label_to_node[label] = node

                # Map generated edges to our nodes
                for edge in generated_edges:
                    source_node_gen = next(
                        (
                            n
                            for n in generated_nodes
                            if n.get("id") == edge.get("source")
                        ),
                        None,
                    )
                    target_node_gen = next(
                        (
                            n
                            for n in generated_nodes
                            if n.get("id") == edge.get("target")
                        ),
                        None,
                    )

                    if source_node_gen and target_node_gen:
                        # Get labels from generated nodes
                        source_label = (
                            source_node_gen.get("data", {})
                            .get("label", "")
                            .lower()
                            .strip()
                        )
                        target_label = (
                            target_node_gen.get("data", {})
                            .get("label", "")
                            .lower()
                            .strip()
                        )

                        # Find matching nodes in our actual nodes
                        source_actual = label_to_node.get(source_label)
                        target_actual = label_to_node.get(target_label)

                        if source_actual and target_actual:
                            edges.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "source": source_actual.get("id"),
                                    "target": target_actual.get("id"),
                                    "type": edge.get("type", "default"),
                                    "animated": edge.get("animated", False),
                                }
                            )
                            logger.info(
                                f"  Mapped edge: {source_label} → {target_label}"
                            )
                        else:
                            if not source_actual:
                                logger.warning(
                                    f"  ⚠️ Could not find source node: {source_label}"
                                )
                            if not target_actual:
                                logger.warning(
                                    f"  ⚠️ Could not find target node: {target_label}"
                                )

                logger.info(f"✅ Mapped {len(edges)} edges to actual node IDs")

                # If we have good edge coverage (at least n-1 edges), return them
                min_edges = len(nodes) - 1
                if edges and len(edges) >= min_edges:
                    logger.info(
                        f"✅ API-generated edges are sufficient ({len(edges)} >= {min_edges})"
                    )
                    return edges
                elif edges:
                    logger.warning(
                        f"⚠️ Only {len(edges)} edges from API, need at least {min_edges}, falling back to similarity"
                    )
                else:
                    logger.warning("⚠️ No edges mapped from API response")

            # Fallback: Use context-aware intelligent edge generation instead of simple similarity
            logger.info("⚠️ Falling back to context-aware intelligent edge generation")
            # Create a basic context analysis from the query
            context_analysis = {
                "workflow_intent": query or "Connect tools in a meaningful workflow",
                "workflow_type": "sequential",
                "key_requirements": [],
            }
            intelligent_edges = await self._generate_intelligent_edges_with_context(
                nodes, context_analysis, query
            )
            if intelligent_edges:
                return intelligent_edges
            # Final fallback: similarity-based edges
            logger.warning(
                "⚠️ Context-aware generation failed, using similarity-based fallback"
            )
            return await self._generate_similarity_based_edges(nodes)

        except Exception as e:
            logger.error(f"❌ Error regenerating edges: {e}", exc_info=True)
            # Final fallback: similarity-based edges
            return await self._generate_similarity_based_edges(nodes)

    async def _generate_similarity_based_edges(self, nodes: List[Dict]) -> List[Dict]:
        """
        Generate SMART edges based on tool functionality similarity.
        IMPROVED: Ensures ALL nodes are connected with intelligent relationships.
        """
        if len(nodes) == 0:
            return []

        if len(nodes) == 1:
            # Single node: no edges needed
            return []

        logger.info(f"🔧 Generating similarity-based edges for {len(nodes)} nodes")

        edges = []

        # Calculate similarity between all pairs of tools
        similarity_scores = []

        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i >= j:  # Skip self and already calculated pairs
                    continue

                # Calculate how similar these tools are
                score = self._calculate_tool_similarity(node_a, node_b)
                similarity_scores.append(
                    {
                        "score": score,
                        "source_idx": i,
                        "target_idx": j,
                        "source_id": node_a["id"],
                        "target_id": node_b["id"],
                        "source_name": node_a.get("data", {}).get("label", "Unknown"),
                        "target_name": node_b.get("data", {}).get("label", "Unknown"),
                    }
                )

        # Sort by similarity (highest first)
        similarity_scores.sort(key=lambda x: x["score"], reverse=True)

        # Track connections per node
        node_connections = {node["id"]: [] for node in nodes}

        # Strategy: Create edges for most similar pairs first
        # Aim for at least n-1 edges to ensure connectivity
        min_edges_needed = len(nodes) - 1

        for pair in similarity_scores:
            source_id = pair["source_id"]
            target_id = pair["target_id"]

            # Add edges for highly similar tools (score > 0)
            # OR if we haven't reached minimum connectivity
            if pair["score"] > 0 or len(edges) < min_edges_needed:
                # Avoid duplicate edges
                edge_exists = any(
                    (e["source"] == source_id and e["target"] == target_id)
                    or (e["source"] == target_id and e["target"] == source_id)
                    for e in edges
                )

                if not edge_exists:
                    edges.append(
                        {
                            "id": str(uuid.uuid4()),
                            "source": source_id,
                            "target": target_id,
                            "type": "default",
                            "animated": False,
                        }
                    )

                    node_connections[source_id].append(target_id)
                    node_connections[target_id].append(source_id)

                    logger.info(
                        f"  Added edge: {pair['source_name']} ↔ {pair['target_name']} (score: {pair['score']:.2f})"
                    )

            # Stop after creating enough edges for good connectivity
            # But ensure each node has at least 1 connection
            if (
                len(edges) >= len(nodes) * 1.5
            ):  # Allow up to 1.5 * n edges for rich connectivity
                break

        # CRITICAL: Ensure EVERY node has at least 1 connection
        logger.info("🔍 Checking for isolated nodes...")
        for node in nodes:
            node_id = node["id"]
            node_name = node.get("data", {}).get("label", "Unknown")

            if len(node_connections[node_id]) == 0:
                # Node is isolated! Connect it to the most similar node
                logger.warning(
                    f"⚠️ Node '{node_name}' is isolated, finding best connection..."
                )

                # Find best match for this isolated node
                best_match = None
                best_score = -1

                for pair in similarity_scores:
                    if (
                        pair["source_id"] == node_id or pair["target_id"] == node_id
                    ) and pair["score"] > best_score:
                        best_score = pair["score"]
                        best_match = pair

                if best_match:
                    # Connect isolated node to best match
                    if best_match["source_id"] == node_id:
                        source, target = (
                            best_match["source_id"],
                            best_match["target_id"],
                        )
                        target_name = best_match["target_name"]
                    else:
                        source, target = (
                            best_match["target_id"],
                            best_match["source_id"],
                        )
                        target_name = best_match["source_name"]

                    # Check if edge already exists
                    edge_exists = any(
                        (e["source"] == source and e["target"] == target)
                        or (e["source"] == target and e["target"] == source)
                        for e in edges
                    )

                    if not edge_exists:
                        edges.append(
                            {
                                "id": str(uuid.uuid4()),
                                "source": source,
                                "target": target,
                                "type": "default",
                                "animated": False,
                            }
                        )
                        logger.info(
                            f"  ✅ Connected isolated node '{node_name}' to '{target_name}' (score: {best_score:.2f})"
                        )

        logger.info(f"✅ Generated {len(edges)} similarity-based edges")
        return edges

    async def _generate_intelligent_edges_with_context(
        self, nodes: List[Dict], context_analysis: Dict[str, Any], user_query: str = ""
    ) -> List[Dict]:
        """
        Generate intelligent edges using LLM with workflow context.
        This creates meaningful connections based on tool functionality and workflow intent.

        Args:
            nodes: List of workflow nodes
            context_analysis: Analysis of conversation context
            user_query: User's query for context

        Returns:
            List of intelligently generated edges
        """
        try:
            if len(nodes) < 2:
                return []

            logger.info(
                f"🧠 Generating intelligent edges with context for {len(nodes)} nodes"
            )

            # Prepare tool information for LLM
            tools_info = []
            for i, node in enumerate(nodes, 1):
                node_data = node.get("data", {})
                tools_info.append(
                    {
                        "id": node.get("id"),
                        "name": node_data.get("label", "Unknown"),
                        "description": node_data.get("description", "")[:200],
                        "features": node_data.get("features", [])[:5],
                        "tags": node_data.get("tags", [])[:5],
                    }
                )

            # Build context from analysis
            workflow_intent = context_analysis.get("workflow_intent", "")
            workflow_type = context_analysis.get("workflow_type", "sequential")
            key_requirements = context_analysis.get("key_requirements", [])

            # Create intelligent prompt
            tools_text = "\n".join(
                [
                    f"{i}. {tool['name']} (ID: {tool['id']}): {tool['description']}\n   Features: {', '.join(tool['features'][:3]) if tool['features'] else 'N/A'}"
                    for i, tool in enumerate(tools_info, 1)
                ]
            )

            requirements_text = (
                ", ".join(key_requirements[:5]) if key_requirements else "Not specified"
            )

            prompt = f"""You are a workflow automation expert. Create intelligent connections between tools based on their functionality and the user's goal.

USER'S GOAL: "{user_query}"

WORKFLOW INTENT: {workflow_intent or "General automation workflow"}
WORKFLOW TYPE: {workflow_type}
KEY REQUIREMENTS: {requirements_text}

TOOLS TO CONNECT:
{tools_text}

TASK: Create logical connections between these tools that make sense for the user's goal.

CONNECTION RULES:
1. Connect tools that logically work together in sequence (e.g., data collection → processing → action)
2. Create connections based on data flow (output of one tool feeds into another)
3. Consider tool categories and complementary functionality
4. Ensure the workflow makes sense for: "{user_query}"
5. Each tool should connect to 1-3 other tools (not all tools need to connect to all others)
6. Avoid circular dependencies unless they represent feedback loops
7. Create a logical flow that achieves the user's goal

Return ONLY valid JSON in this format:
{{
    "edges": [
        {{
            "source": "node_id_1",
            "target": "node_id_2",
            "reason": "Brief explanation of why these tools connect"
        }}
    ]
}}

IMPORTANT:
- Use the exact node IDs from the tools list above
- Only create connections that make logical sense
- Focus on connections that help achieve: "{user_query}"
- Return at least {len(nodes) - 1} edges to ensure connectivity
- Maximum {len(nodes) * 2} edges to avoid over-connecting"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if result and "edges" in result:
                edges = []
                node_ids = {node.get("id") for node in nodes}

                for edge_data in result["edges"]:
                    source_id = edge_data.get("source")
                    target_id = edge_data.get("target")

                    # Validate both nodes exist and no self-loops
                    if (
                        source_id in node_ids
                        and target_id in node_ids
                        and source_id != target_id
                    ):
                        # Check for duplicates
                        edge_exists = any(
                            (e["source"] == source_id and e["target"] == target_id)
                            or (e["source"] == target_id and e["target"] == source_id)
                            for e in edges
                        )

                        if not edge_exists:
                            edges.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "source": source_id,
                                    "target": target_id,
                                    "type": "default",
                                    "animated": False,
                                }
                            )
                            logger.debug(
                                f"✅ Intelligent edge: {edge_data.get('reason', 'No reason provided')}"
                            )

                logger.info(f"✅ Generated {len(edges)} intelligent context-aware edges")
                return edges
            else:
                logger.warning(
                    "LLM did not return valid edges, falling back to similarity"
                )
                return await self._generate_similarity_based_edges(nodes)

        except Exception as e:
            logger.error(f"Error generating intelligent edges: {e}", exc_info=True)
            # Fallback to similarity-based
            return await self._generate_similarity_based_edges(nodes)

    def _calculate_tool_similarity(self, node_a: Dict, node_b: Dict) -> float:
        """Calculate similarity between two tools based on their data."""
        try:
            data_a = node_a.get("data", {})
            data_b = node_b.get("data", {})

            # Get text data
            desc_a = data_a.get("description", "").lower()
            desc_b = data_b.get("description", "").lower()
            features_a = str(data_a.get("features", "")).lower()
            features_b = str(data_b.get("features", "")).lower()
            tags_a = str(data_a.get("tags", "")).lower()
            tags_b = str(data_b.get("tags", "")).lower()

            # Combine all text
            text_a = f"{desc_a} {features_a} {tags_a}"
            text_b = f"{desc_b} {features_b} {tags_b}"

            # Keywords that indicate tool relationships
            keywords = {
                "crm": ["sales", "customer", "lead", "contact", "pipeline"],
                "email": ["email", "campaign", "newsletter", "mail"],
                "social": [
                    "social",
                    "instagram",
                    "facebook",
                    "twitter",
                    "linkedin",
                    "post",
                ],
                "content": ["content", "writing", "blog", "article", "text"],
                "automation": ["automation", "workflow", "automate", "schedule"],
                "analytics": ["analytics", "report", "metrics", "data", "insights"],
                "design": ["design", "image", "graphic", "visual", "photo"],
                "video": ["video", "film", "movie", "clip"],
            }

            # Find common categories
            score = 0.0
            for category, words in keywords.items():
                in_a = any(word in text_a for word in words)
                in_b = any(word in text_b for word in words)
                if in_a and in_b:
                    score += 1.0

            # Common words boost
            words_a = set(text_a.split())
            words_b = set(text_b.split())
            common_words = words_a & words_b
            score += len(common_words) * 0.1

            return score

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def _optimize_node_positions(self, nodes: List[Dict]) -> List[Dict]:
        """Optimize node positions for better visualization."""
        try:
            positioned_nodes = []

            # Calculate positions based on node count
            x_spacing = 250
            y_spacing = 150
            start_x = 100
            start_y = 100

            for i, node in enumerate(nodes):
                # Create a copy to avoid mutating original
                positioned_node = node.copy()

                # Calculate position (left to right flow)
                row = i // 3  # 3 nodes per row
                col = i % 3

                positioned_node["position"] = {
                    "x": start_x + (col * x_spacing),
                    "y": start_y + (row * y_spacing),
                }

                positioned_nodes.append(positioned_node)

            return positioned_nodes

        except Exception as e:
            logger.error(f"Error optimizing positions: {e}")
            # Return nodes as-is
            return nodes

    async def _save_workflow_to_database(
        self,
        workflow_data: Dict[str, Any],
        conversation_session,
        request_user,
        workflow_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        """Save or update the generated workflow in the database."""
        try:
            from workflow.models import Workflow

            # Use provided workflow_id or create new one
            if workflow_id:
                logger.info(f"🔄 Updating existing workflow with ID: {workflow_id}")

                # Try to get existing workflow
                try:
                    workflow_obj = await sync_to_async(Workflow.objects.get)(
                        id=workflow_id
                    )

                    # Update existing workflow
                    workflow_obj.metadata = workflow_data
                    workflow_obj.user_query = conversation_session.original_query
                    workflow_obj.prompt = conversation_session.original_query
                    await sync_to_async(workflow_obj.save)()

                    logger.info(f"✅ Updated existing workflow: {workflow_id}")

                    # Also save to Node and Edge tables
                    await self._save_to_node_edge_tables(workflow_obj, workflow_data)

                    return workflow_id

                except Workflow.DoesNotExist:
                    logger.warning(
                        f"⚠️ Workflow {workflow_id} not found, creating new one"
                    )
                    # Create new workflow with the provided ID
                    workflow_obj = await sync_to_async(Workflow.objects.create)(
                        id=workflow_id,
                        name=f"Workflow from conversation - {conversation_session.original_query[:50]}",
                        metadata=workflow_data,
                        owner=request_user,
                        user_query=conversation_session.original_query,
                        prompt=conversation_session.original_query,
                    )
                    logger.info(
                        f"✅ Created new workflow with provided ID: {workflow_id}"
                    )

                    # Also save to Node and Edge tables
                    await self._save_to_node_edge_tables(workflow_obj, workflow_data)

                    return workflow_id
            else:
                # Create new workflow with generated ID
                new_workflow_id = uuid.uuid4()
                logger.info(f"🔄 Creating new workflow with ID: {new_workflow_id}")

                workflow_obj = await sync_to_async(Workflow.objects.create)(
                    id=new_workflow_id,
                    name=f"Workflow from conversation - {conversation_session.original_query[:50]}",
                    metadata=workflow_data,
                    owner=request_user,
                    user_query=conversation_session.original_query,
                    prompt=conversation_session.original_query,
                )

                logger.info(f"✅ Created new workflow: {new_workflow_id}")

                # Also save to Node and Edge tables
                await self._save_to_node_edge_tables(workflow_obj, workflow_data)

                return new_workflow_id

        except Exception as e:
            logger.error(f"❌ Error saving workflow to database: {e}", exc_info=True)
            # Return the provided workflow_id or generate fallback
            fallback_id = workflow_id if workflow_id else uuid.uuid4()
            logger.warning(f"⚠️ Returning fallback workflow ID: {fallback_id}")
            return fallback_id

    async def _save_to_node_edge_tables(
        self, workflow_obj, workflow_data: Dict[str, Any]
    ):
        """Save workflow nodes and edges to separate Node and Edge tables.

        Since nodes now use UUIDs from the start, we can use them directly as database IDs.
        No ID mapping needed!
        """
        try:
            from workflow.models import Edge, Node

            nodes = workflow_data.get("nodes", [])
            edges = workflow_data.get("edges", [])

            logger.info(
                f"💾 Saving {len(nodes)} nodes and {len(edges)} edges to Node/Edge tables"
            )

            # Delete existing nodes and edges for this workflow
            await sync_to_async(
                Node.objects.filter(workflow_id=workflow_obj.id).delete
            )()
            await sync_to_async(
                Edge.objects.filter(workflow_id=workflow_obj.id).delete
            )()

            # Save nodes (using their existing UUIDs with proper conflict handling)
            for node_data in nodes:
                node_uuid = node_data.get("id")
                if not node_uuid:
                    node_uuid = str(uuid.uuid4())
                    logger.warning(
                        f"⚠️ Node missing ID, generated new UUID: {node_uuid}"
                    )

                node_content = (
                    node_data.get("data", {}).copy() if node_data.get("data") else {}
                )

                try:
                    # Use update_or_create to handle existing nodes gracefully
                    node_obj, created = await sync_to_async(
                        Node.objects.update_or_create
                    )(
                        id=node_uuid,
                        workflow=workflow_obj,
                        defaults={
                            "type": node_data.get("type", "default"),
                            "position_x": node_data.get("position", {}).get("x", 0),
                            "position_y": node_data.get("position", {}).get("y", 0),
                            "data": node_content,
                            "dragging": node_data.get("dragging", False),
                            "height": node_data.get("height"),
                            "width": node_data.get("width"),
                            "position_absolute_x": node_data.get(
                                "positionAbsolute", {}
                            ).get("x", 0),
                            "position_absolute_y": node_data.get(
                                "positionAbsolute", {}
                            ).get("y", 0),
                            "selected": node_data.get("selected", False),
                        },
                    )

                    action = "created" if created else "updated"
                    tool_name = node_content.get("label", "Unknown")
                    logger.info(f"✅ Node {action}: {node_uuid} ({tool_name})")

                except Exception as node_error:
                    logger.error(f"❌ Error saving node {node_uuid}: {node_error}")
                    # Generate new UUID and try again
                    new_uuid = str(uuid.uuid4())
                    logger.info(f"🔄 Retrying with new UUID: {new_uuid}")

                    try:
                        await sync_to_async(Node.objects.create)(
                            id=new_uuid,
                            type=node_data.get("type", "default"),
                            position_x=node_data.get("position", {}).get("x", 0),
                            position_y=node_data.get("position", {}).get("y", 0),
                            data=node_content,
                            workflow=workflow_obj,
                            dragging=node_data.get("dragging", False),
                            height=node_data.get("height"),
                            width=node_data.get("width"),
                            position_absolute_x=node_data.get(
                                "positionAbsolute", {}
                            ).get("x", 0),
                            position_absolute_y=node_data.get(
                                "positionAbsolute", {}
                            ).get("y", 0),
                            selected=node_data.get("selected", False),
                        )
                        logger.info(f"✅ Node created with new UUID: {new_uuid}")

                        # Update the node_data with new UUID for edge references
                        node_data["id"] = new_uuid

                    except Exception as retry_error:
                        logger.error(
                            f"❌ Failed to save node even with new UUID: {retry_error}"
                        )
                        continue

            logger.info(f"✅ Saved {len(nodes)} nodes to Node table")

            # Save edges (source and target are already UUIDs)
            for edge_data in edges:
                source_id = edge_data.get("source")
                target_id = edge_data.get("target")

                # Save edge with UUIDs directly (no mapping needed)
                if source_id and target_id:
                    await sync_to_async(Edge.objects.create)(
                        id=str(uuid.uuid4()),
                        source=source_id,  # Already a UUID
                        target=target_id,  # Already a UUID
                        type=edge_data.get("type", "default"),
                        workflow=workflow_obj,
                    )

            logger.info(f"✅ Saved {len(edges)} edges to Edge table")

        except Exception as e:
            logger.error(f"❌ Error saving to Node/Edge tables: {e}", exc_info=True)

    async def _generate_response_message(
        self,
        validated_nodes: List[Dict],
        rejected_nodes: List[Dict],
        edges: List[Dict],
        context_analysis: Dict[str, Any],
    ) -> str:
        """Generate dynamic LLM response for workflow generation."""
        try:
            # Extract tool information for LLM
            tool_names = [
                node.get("data", {}).get("label", "Tool")
                for node in validated_nodes[:5]
            ]
            tool_descriptions = [
                node.get("data", {}).get("description", "")[:100]
                for node in validated_nodes[:3]
            ]

            # Create detailed prompt for LLM with connection info
            prompt = f"""
            The user has successfully generated a workflow with:
            - {len(validated_nodes)} tools: {', '.join(tool_names)}
            - {len(edges)} intelligent connections based on tool functionality
            - Purpose: {context_analysis.get('summary', 'General automation')}

            Tool descriptions:
            {chr(10).join([f"- {name}: {desc}..." for name, desc in zip(tool_names[:3], tool_descriptions)])}

            Generate a warm, professional, and DETAILED response that:
            1. Congratulates them on their new workflow
            2. Mentions specific tools BY NAME and explains what each does
            3. Explains HOW the tools are connected intelligently (e.g., "CRM connects to Email Marketing for lead nurturing")
            4. Describes the workflow logic and data flow between tools
            5. Highlights the automation benefits and efficiency gains
            6. Shows genuine enthusiasm about their automation potential
            7. Suggests next steps naturally

            IMPORTANT:
            - Be conversational and friendly like a professional consultant
            - Be VERY SPECIFIC about the tools and connections
            - ABSOLUTELY NO emojis or special characters (🎉 ✨ 📊 etc.) - plain text only
            - Use natural, flowing language - write like you're talking to a colleague
            - Keep it engaging and informative (4-6 sentences)
            - Sound human, not robotic or overly enthusiastic
            """

            # Use the correct method: generate_response
            message = await self.llm.generate_response(prompt)

            # Add workflow stats
            message += f"\n\nWorkflow Stats:\n"
            message += f"- {len(validated_nodes)} tools configured\n"
            message += f"- {len(edges)} intelligent connections\n"

            if rejected_nodes:
                message += f"- {len(rejected_nodes)} tools filtered out\n"

            message += f"\nReady to automate your work."

            return message

        except Exception as e:
            logger.error(f"Error generating LLM response message: {e}")
            # Fallback to a nice default message
            return (
                f"Your workflow is ready.\n\n"
                f"I've created an intelligent workflow with {len(validated_nodes)} tools "
                f"connected through {len(edges)} smart pathways. These tools work together "
                f"to automate your processes efficiently.\n\n"
                f"Workflow Summary:\n"
                f"- {len(validated_nodes)} tools configured\n"
                f"- {len(edges)} intelligent connections\n\n"
                f"Your workflow is ready to use. You can add more tools or customize it as needed."
            )

    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        """Format chat history for LLM analysis."""
        formatted = []
        for msg in chat_history:
            formatted.append(f"User: {msg.get('user', '')}")
            formatted.append(
                f"AI: {msg.get('ai', '')[:200]}..."
            )  # Truncate long messages
        return "\n".join(formatted)

    def _format_nodes_for_analysis(self, nodes: List[Dict]) -> str:
        """Format nodes for LLM analysis."""
        formatted = []
        for i, node in enumerate(nodes, 1):
            node_data = node.get("data", {})
            formatted.append(
                f"{i}. ID: {node.get('id')}, "
                f"Label: {node_data.get('label', 'Unknown')}, "
                f"Description: {node_data.get('description', 'No description')[:100]}"
            )
        return "\n".join(formatted)

    async def _generate_workflow_from_chat_only(
        self,
        conversation_session,
        request_user,
        workflow_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Generate workflow from chat history when no nodes exist.

        This method:
        1. Analyzes chat to understand what user needs
        2. Searches for relevant tools
        3. Creates nodes from found tools
        4. Generates edges
        5. Saves complete workflow

        Args:
            conversation_session: ConversationSession model instance
            request_user: User making the request
            workflow_id: Optional existing workflow ID to update
        """
        try:
            logger.info("🔍 Analyzing chat history to find relevant tools")

            # Step 1: Extract tool requirements from chat
            tool_requirements = await self._extract_tool_requirements_from_chat(
                conversation_session.chat_history, conversation_session.original_query
            )

            if not tool_requirements.get("queries"):
                return {
                    "status": "error",
                    "message": "I couldn't understand what tools you need from our conversation. Could you be more specific about what you want to build?",
                    "workflow": None,
                    "workflow_id": None,
                }

            logger.info(
                f"📋 Extracted {len(tool_requirements.get('queries', []))} tool requirements"
            )

            # Step 2: Search for tools based on requirements
            all_tools = []
            for query in tool_requirements.get("queries", [])[
                :5
            ]:  # Limit to 5 searches
                logger.info(f"🔎 Searching for: {query}")
                search_result = await self.recommender.search_tools(
                    query=query,
                    max_results=3,
                    include_pinecone=True,
                    include_internet=True,
                )

                if search_result.get("status") == "success":
                    tools = search_result.get("tools", [])
                    all_tools.extend(tools[:2])  # Take top 2 from each search

            if not all_tools:
                return {
                    "status": "error",
                    "message": "I couldn't find any tools matching our conversation. Try being more specific about what you need.",
                    "workflow": None,
                    "workflow_id": None,
                }

            logger.info(f"✅ Found {len(all_tools)} tools total")

            # Step 3: Create nodes from tools using helper method
            workflow_nodes = []
            for i, tool in enumerate(all_tools[:10], 1):  # Limit to 10 tools
                node = self._create_tool_node_from_dict(tool, i)
                workflow_nodes.append(node)

            # Step 4: Generate edges using helper method
            workflow_type = tool_requirements.get("workflow_type", "sequential")

            if workflow_type == "sequential" and len(workflow_nodes) > 1:
                # Use helper method to generate similarity-based edges
                edges = await self._generate_similarity_based_edges(workflow_nodes)
            else:
                edges = []

            logger.info(f"🔗 Generated {len(edges)} edges")

            # Step 5: Create workflow structure
            workflow_data = {
                "nodes": workflow_nodes,
                "edges": edges,
                "metadata": {
                    "generated_from": "chat_history",
                    "original_query": conversation_session.original_query,
                    "tool_requirements": tool_requirements.get("queries", []),
                    "generation_method": "ai_search_and_build",
                    "auto_generated": True,
                },
            }

            # Step 6: Save to database
            saved_workflow_id = await self._save_workflow_to_database(
                workflow_data, conversation_session, request_user, workflow_id
            )

            # Step 7: Update conversation session
            conversation_session.workflow_nodes = workflow_nodes
            conversation_session.workflow_edges = edges
            await sync_to_async(conversation_session.save)()

            # Step 8: Generate message
            message = (
                f"I've generated your workflow from our conversation.\n\n"
                f"Based on what you told me, I found and added {len(workflow_nodes)} tools "
                f"with {len(edges)} connections.\n\n"
                f"What I understood you need:\n"
            )

            for query in tool_requirements.get("queries", [])[:3]:
                message += f"- {query}\n"

            message += (
                f"\nNext steps:\n"
                f"- Review the tools I selected\n"
                f"- Add or remove tools as needed\n"
                f"- Customize the connections\n"
                f"- Execute your workflow"
            )

            logger.info(
                f"✅ Successfully generated workflow from chat: {saved_workflow_id}"
            )

            return {
                "status": "success",
                "workflow": workflow_data,
                "workflow_id": str(saved_workflow_id),
                "message": message,
                "metadata": {
                    "total_nodes": len(workflow_nodes),
                    "total_edges": len(edges),
                    "rejected_nodes": 0,
                    "generated_from": "chat_history",
                    "tool_requirements": tool_requirements.get("queries", []),
                },
            }

        except Exception as e:
            logger.error(f"❌ Error generating workflow from chat: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to generate workflow from chat: {str(e)}",
                "workflow": None,
                "workflow_id": None,
            }

    async def _extract_tool_requirements_from_chat(
        self, chat_history: List[Dict], original_query: str
    ) -> Dict[str, Any]:
        """Extract what tools the user needs from chat history."""
        try:
            # Prepare chat context
            chat_context = self._format_chat_history(
                chat_history[-10:]
            )  # Last 10 messages

            extraction_prompt = f"""
            Analyze this conversation and determine what tools/software the user needs.

            Original Query: {original_query}

            Chat History:
            {chat_context}

            Extract:
            1. What specific tools or types of tools does the user need?
            2. What is the workflow type? (sequential, parallel, etc.)
            3. What are the main use cases?

            Return JSON with:
            {{
                "queries": [
                    "specific search query for tool 1",
                    "specific search query for tool 2",
                    ...
                ],
                "workflow_type": "sequential" or "parallel",
                "use_cases": ["use case 1", "use case 2"]
            }}

            Generate 2-5 specific search queries that would find the right tools.
            Be specific - instead of "CRM", say "CRM for sales automation".

            Return ONLY valid JSON.
            """

            response = await self.llm.generate_response(extraction_prompt)
            result = await self.llm.parse_json_response(response)

            if not result:
                # Fallback: use original query
                return {
                    "queries": [original_query],
                    "workflow_type": "sequential",
                    "use_cases": [],
                }

            return result

        except Exception as e:
            logger.error(f"Error extracting requirements: {e}")
            return {
                "queries": [original_query],
                "workflow_type": "sequential",
                "use_cases": [],
            }

    async def generate_workflow_from_questionnaire(
        self,
        conversation_session,
        request_user,
        workflow_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """
        Generate workflow from completed questionnaire using intelligent search.

        This method is called automatically when all questions are answered.
        It uses the NEW INTELLIGENT PIPELINE:
        1. Extract refined query (comprehensive problem analysis)
        2. Generate intelligent search queries (paragraph-level descriptions)
        3. Execute parallel searches for each query
        4. Intelligently select optimal tools based on refined query
        5. Generate comprehensive workflow
        6. Save to database
        7. Return to user

        Args:
            conversation_session: ConversationSession with completed questionnaire
            request_user: User making the request
            workflow_id: Optional existing workflow ID to update

        Returns:
            Dict with workflow, workflow_id, message, etc.
        """
        try:
            logger.info(
                f"🚀 Generating workflow from questionnaire for session {conversation_session.session_id}"
            )
            logger.info("🧠 Using NEW INTELLIGENT PIPELINE with refined query analysis")

            # Import services
            from .questionnaire_service import QuestionnaireService

            questionnaire_service = QuestionnaireService()

            questionnaire_json = conversation_session.questionnaire_json

            # Verify questionnaire is complete
            if not questionnaire_service.is_questionnaire_complete(questionnaire_json):
                logger.error("❌ Cannot generate workflow - questionnaire not complete")
                return {
                    "status": "error",
                    "message": "Cannot generate workflow yet. Please answer all required questions first.",
                    "workflow": None,
                    "workflow_id": None,
                }

            # STEP 1: Extract refined query (comprehensive analysis)
            workflow_info = questionnaire_json.get("workflow_info", {})
            refined_query = workflow_info.get("refined_query")

            if not refined_query:
                # Fallback to building from answers
                logger.warning("⚠️ No refined query found, building from answers")
                refined_query = questionnaire_service.build_enriched_search_query(
                    questionnaire_json
                )

            # Count total answers across all phases
            total_answers = 0
            for phase_num in [1, 2, 3]:
                phase_key = f"phase_{phase_num}"
                phase_data = questionnaire_json.get(phase_key, {})
                total_answers += len(phase_data.get("answers", {}))

            logger.info(
                f"📝 Using refined query from {total_answers} answers across 3 phases"
            )
            logger.info(f"📄 Refined query length: {len(refined_query)} characters")

            # STEP 2: Generate intelligent search queries from refined query
            logger.info("🎯 Generating intelligent search queries from refined query...")
            from .services.intelligent_query_generator import IntelligentQueryGenerator

            query_generator = IntelligentQueryGenerator()
            query_generation_result = await query_generator.generate_search_queries(
                refined_query=refined_query,
                original_query=conversation_session.original_query,
                max_queries=4,  # Generate up to 4 detailed search queries
            )

            if query_generation_result.get("status") != "success":
                logger.error("❌ Search query generation failed")
                return {
                    "status": "error",
                    "message": "Failed to generate search queries. Please try again.",
                    "workflow": None,
                    "workflow_id": None,
                }

            search_queries = query_generation_result.get("search_queries", [])
            workflow_understanding = query_generation_result.get(
                "workflow_understanding", ""
            )

            logger.info(f"✅ Generated {len(search_queries)} intelligent search queries")
            for i, sq in enumerate(search_queries[:3], 1):
                logger.info(
                    f"   Query {i} ({sq.get('category')}): {sq.get('query')[:80]}..."
                )

            # Step 3: Execute intelligent search with Pinecone-first approach
            logger.info(
                f"🔎 executing {len(search_queries)} intelligent searches in PARALLEL..."
            )
            import asyncio

            async def execute_search(query_obj):
                try:
                    query_text = query_obj.get("query", "")
                    category = query_obj.get("category", "General")
                    priority = query_obj.get("priority", "medium")

                    if not query_text:
                        return []

                    logger.info(f"   🚀 Starting search for category: {category}")

                    # Enhance query slightly for Pinecone
                    search_result = (
                        await questionnaire_service.recommender.search_tools(
                            query=query_text,
                            max_results=8,  # Get up to 8 candidates per query
                            include_pinecone=True,
                            include_internet=True,  # Enable internet search for gaps
                        )
                    )

                    if search_result.get("status") == "success":
                        tools = search_result.get("tools", [])
                        # Tag tools with the category they were found for
                        for tool in tools:
                            tool["_found_for_category"] = category
                            tool["_found_for_priority"] = priority
                            tool["_search_query"] = query_text
                        logger.info(
                            f"   ✅ Search for '{category}' returned {len(tools)} tools"
                        )
                        return tools
                    return []
                except Exception as e:
                    logger.error(f"   ❌ Search failed for query: {e}")
                    return []

            # Execute all searches in parallel
            search_tasks = [execute_search(q) for q in search_queries]
            search_results_lists = await asyncio.gather(*search_tasks)

            # Flatten results
            all_found_tools = []
            for result_list in search_results_lists:
                all_found_tools.extend(result_list)

            # Deduplicate and validate found tools
            seen_ids = set()
            deduplicated_tools = []

            for tool in all_found_tools:
                # Use a unique identifier (url or title)
                uid = (
                    tool.get("url") or tool.get("link") or tool.get("Title", "").lower()
                )
                if uid and uid not in seen_ids:
                    seen_ids.add(uid)
                    deduplicated_tools.append(tool)

            search_metadata = {
                "pinecone_count": len(
                    [
                        t
                        for t in deduplicated_tools
                        if t.get("Source") == "Pinecone Vector Database"
                    ]
                ),
                "internet_count": len(
                    [
                        t
                        for t in deduplicated_tools
                        if t.get("Source") != "Pinecone Vector Database"
                    ]
                ),
                "skipped_internet": False,
            }

            logger.info(
                f"✅ Intelligent search complete: {len(deduplicated_tools)} tools found"
            )
            logger.info(
                f"   Pinecone: {search_metadata.get('pinecone_count', 0)} tools"
            )
            logger.info(
                f"   Internet: {search_metadata.get('internet_count', 0)} tools"
            )
            logger.info(
                f"   Skipped internet: {search_metadata.get('skipped_internet', False)}"
            )

            if not deduplicated_tools:
                logger.error("❌ No tools found after search")
                return {
                    "status": "error",
                    "message": "I couldn't find any tools matching your requirements. Try adjusting your answers.",
                    "workflow": None,
                    "workflow_id": None,
                }

            # STEP 4: Intelligently select optimal tools based on refined query
            logger.info("🎯 Selecting optimal tools using refined query analysis...")
            from .services.refined_query_tool_selector import RefinedQueryToolSelector

            tool_selector = RefinedQueryToolSelector()
            selection_result = await tool_selector.select_optimal_tools(
                candidate_tools=deduplicated_tools,
                refined_query=refined_query,
                workflow_understanding=workflow_understanding,
                max_tools=10,  # Select up to 10 optimal tools
            )

            if selection_result.get("status") != "success":
                logger.error("❌ Tool selection failed")
                return {
                    "status": "error",
                    "message": "Failed to select optimal tools. Please try again.",
                    "workflow": None,
                    "workflow_id": None,
                }

            selected_tools = selection_result.get("selected_tools", [])
            logger.info(f"✅ Selected {len(selected_tools)} optimal tools")
            logger.info(
                f"   Confidence: {selection_result.get('confidence_score', 0):.2f}"
            )
            logger.info(
                f"   Selection reasoning: {selection_result.get('selection_reasoning', '')[:100]}..."
            )

            tools = selected_tools  # Use selected tools for workflow generation
            logger.info(f"✅ Found {len(tools)} tools from search")

            # 🚀 Submit internet-discovered tools for background scraping
            internet_tools = [
                t for t in tools if "Internet Search" in t.get("Source", "")
            ]
            if internet_tools:
                logger.info(
                    f"🚀 Submitting {len(internet_tools)} internet-discovered tools for scraping"
                )
                try:
                    from .internet_tool_submitter import internet_tool_submitter

                    submission_result = (
                        await internet_tool_submitter.submit_tools_async(
                            internet_tools, source_query=refined_query
                        )
                    )
                    if submission_result:
                        logger.info(
                            f"✅ Tools submitted for scraping. Job ID: {submission_result.get('id')}"
                        )
                    else:
                        logger.warning("⚠️ Failed to submit tools for scraping")
                except Exception as submit_error:
                    logger.error(
                        f"⚠️ Error submitting tools for scraping: {submit_error}"
                    )
                    # Don't fail workflow generation if submission fails

            if not tools:
                return {
                    "status": "error",
                    "message": "I couldn't find any tools matching your requirements. Try adjusting your answers.",
                    "workflow": None,
                    "workflow_id": None,
                }

            # STEP 5: Generate workflow using sequential workflow generator
            logger.info("🔨 Generating sequential workflow structure...")
            from .services.sequential_workflow_generator import (
                SequentialWorkflowGenerator,
            )

            workflow_generator = SequentialWorkflowGenerator()
            workflow = await workflow_generator.generate_sequential_workflow(
                tools=tools,
                refined_query=refined_query,
                original_query=conversation_session.original_query,
            )

            if not workflow or not workflow.get("nodes"):
                logger.error("❌ Workflow generation failed")
                return {
                    "status": "error",
                    "message": "Failed to generate workflow. Please try again.",
                    "workflow": None,
                    "workflow_id": None,
                }

            logger.info(
                f"✅ Generated workflow with {len(workflow.get('nodes', []))} nodes"
            )

            # Add metadata about questionnaire
            if "metadata" not in workflow:
                workflow["metadata"] = {}

            workflow["metadata"][
                "generated_from"
            ] = "questionnaire_intelligent_pipeline"
            workflow["metadata"]["questionnaire_answers"] = questionnaire_json.get(
                "answers", {}
            )
            workflow["metadata"]["original_query"] = conversation_session.original_query
            workflow["metadata"]["refined_query"] = refined_query
            workflow["metadata"]["workflow_understanding"] = workflow_understanding
            workflow["metadata"]["search_queries_used"] = len(search_queries)
            workflow["metadata"]["candidate_tools_found"] = len(deduplicated_tools)
            workflow["metadata"]["tools_selected"] = len(selected_tools)
            workflow["metadata"]["selection_confidence"] = selection_result.get(
                "confidence_score", 0
            )

            # Save workflow to database
            logger.info("💾 Saving workflow to database...")
            saved_workflow_id = await self._save_workflow_to_database(
                workflow, conversation_session, request_user, workflow_id
            )

            # Update conversation session with workflow
            conversation_session.workflow_nodes = workflow.get("nodes", [])
            conversation_session.workflow_edges = workflow.get("edges", [])
            await sync_to_async(conversation_session.save)()

            # Save RefinedQuery to separate table (update if exists, create if not)
            logger.info("💾 Saving refined query to database...")
            from .models import RefinedQuery

            workflow_info = questionnaire_json.get("workflow_info", {})
            refined_query_text = workflow_info.get("refined_query", refined_query)

            # Use update_or_create to update existing or create new
            refined_query_obj, created = await sync_to_async(
                RefinedQuery.objects.update_or_create
            )(
                workflow_id=saved_workflow_id,
                defaults={
                    "user": request_user,
                    "session": conversation_session,
                    "original_query": conversation_session.original_query,
                    "refined_query": refined_query_text,
                    "workflow_info": workflow_info,
                },
            )

            action = "created" if created else "updated"
            logger.info(
                f"✅ Refined query {action} for workflow ID: {saved_workflow_id}"
            )

            logger.info(f"✅ Workflow saved with ID: {saved_workflow_id}")

            # Generate success message
            message = await self._generate_questionnaire_success_message(
                workflow, questionnaire_json, conversation_session.original_query
            )

            return {
                "status": "success",
                "workflow": workflow,
                "workflow_id": str(saved_workflow_id),
                "message": message,
                "metadata": {
                    "total_nodes": len(workflow.get("nodes", [])),
                    "total_edges": len(workflow.get("edges", [])),
                    "generated_from": "questionnaire",
                    "questions_answered": total_answers,
                },
            }

        except Exception as e:
            logger.error(
                f"❌ Error generating workflow from questionnaire: {e}", exc_info=True
            )
            return {
                "status": "error",
                "message": f"An error occurred while generating your workflow: {str(e)}",
                "workflow": None,
                "workflow_id": None,
            }

    async def _generate_questionnaire_success_message(
        self,
        workflow: Dict[str, Any],
        questionnaire_json: Dict[str, Any],
        original_query: str,
    ) -> str:
        """Generate a success message for questionnaire-based workflow."""
        try:
            nodes = workflow.get("nodes", [])
            edges = workflow.get("edges", [])

            # Get tool names
            tool_names = [
                node.get("data", {}).get("label", "Tool") for node in nodes[:5]
            ]

            # Count total answers across all phases
            total_answers = 0
            for phase_num in [1, 2, 3]:
                phase_key = f"phase_{phase_num}"
                phase_data = questionnaire_json.get(phase_key, {})
                total_answers += len(phase_data.get("answers", {}))

            # Generate a natural, human message without emojis
            tool_list = ", ".join(tool_names[:3])
            if len(tool_names) > 3:
                tool_list += f", and {len(tool_names) - 3} more"

            message = (
                f"Thanks for completing the questionnaire. Based on your answers, "
                f"I've put together a workflow with {len(nodes)} tools that should work well for your needs.\n\n"
                f"The workflow includes {tool_list}. "
            )

            if len(edges) > 0:
                message += f"I've set up {len(edges)} connections between them so data flows automatically where it makes sense. "

            message += (
                f"Feel free to review everything and make any adjustments. "
                f"You can add more tools, remove ones you don't need, or customize how they connect.\n\n"
                f"Workflow Summary:\n"
                f"  - {len(nodes)} tools configured\n"
                f"  - {len(edges)} connections set up\n"
                f"  - {total_answers} questions answered\n\n"
                f"Your workflow is ready to use whenever you are."
            )

            return message

        except Exception as e:
            logger.error(f"Error generating success message: {e}")
            nodes = workflow.get("nodes", [])
            edges = workflow.get("edges", [])
            # Count total answers in fallback
            total_answers_fallback = 0
            for phase_num in [1, 2, 3]:
                phase_key = f"phase_{phase_num}"
                phase_data = questionnaire_json.get(phase_key, {})
                total_answers_fallback += len(phase_data.get("answers", {}))

            return (
                f"Thank you for completing the questionnaire. Based on your {total_answers_fallback} answers, "
                f"I've created a workflow with {len(nodes)} tools and {len(edges)} intelligent connections. "
                f"These tools will work together seamlessly to automate your processes.\n\n"
                f"Workflow Summary:\n"
                f"- {len(nodes)} tools configured\n"
                f"- {len(edges)} intelligent connections\n"
                f"- {total_answers_fallback} questions answered\n\n"
                f"Your workflow is ready. Take a moment to review and customize it to your needs."
            )

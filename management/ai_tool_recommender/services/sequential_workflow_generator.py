"""Sequential Workflow Generator - Creates workflows with proper execution sequence.

This service generates workflows that follow a logical execution sequence,
not just connecting all tools randomly. It understands the flow of:
Input → Process → Action → Monitor → Cleanup/Return
"""

import logging
import uuid
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class SequentialWorkflowGenerator:
    """Service that generates workflows with sequential execution logic."""

    def __init__(self):
        """Initialize the sequential workflow generator."""
        self.llm = get_shared_llm()

    def _get_simple_stage(self, step: int, total: int) -> str:
        """Get a simple stage name based on position in workflow."""
        if step == 1:
            return "start"
        elif step == total:
            return "end"
        else:
            return "process"

    async def _enrich_tools_with_problem_solving(
        self, tools: List[Dict[str, Any]], refined_query: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich tools (especially internet tools) with analysis of how they solve the user's problem.

        Args:
            tools: List of tools
            refined_query: User's problem description

        Returns:
            Enriched tools with problem-solving analysis
        """
        try:
            enriched_tools = []

            for tool in tools:
                # Only enrich internet tools (Pinecone tools already have good data)
                source = tool.get("Source", "")
                if "Internet" in source:
                    logger.info(
                        f"🔍 Enriching internet tool: {tool.get('Title', 'Unknown')}"
                    )

                    # Use LLM to analyze how this tool solves the problem
                    prompt = f"""Analyze how this tool helps solve the user's problem.

USER'S PROBLEM:
{refined_query[:500]}

TOOL INFORMATION:
- Name: {tool.get('Title', 'Unknown')}
- Description: {tool.get('Description', 'N/A')[:300]}
- Features: {tool.get('Features', 'N/A')}
- Website: {tool.get('Website', 'N/A')}

YOUR TASK:
Explain in 2-3 sentences:
1. What specific problem does this tool solve?
2. How does it help achieve the user's goal?
3. What role should it play in the workflow?

Be specific and practical. Focus on the tool's capabilities and how they address the user's needs.

Response (2-3 sentences):"""

                    try:
                        analysis = await self.llm.generate_response(prompt)
                        # Add enrichment to tool data
                        tool["problem_solving_analysis"] = analysis.strip()
                        logger.info(f"✅ Enriched: {tool.get('Title', 'Unknown')}")
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Failed to enrich tool {tool.get('Title', 'Unknown')}: {e}"
                        )
                        tool[
                            "problem_solving_analysis"
                        ] = f"Helps with {tool.get('Title', 'Unknown')} functionality"
                else:
                    # Pinecone tools - use existing description
                    tool["problem_solving_analysis"] = tool.get("Description", "")[:200]

                enriched_tools.append(tool)

            return enriched_tools

        except Exception as e:
            logger.error(f"❌ Error enriching tools: {e}")
            return tools  # Return original tools if enrichment fails

    async def generate_sequential_workflow(
        self,
        tools: List[Dict[str, Any]],
        refined_query: str,
        original_query: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a workflow with proper sequential execution flow.

        Args:
            tools: List of selected tools
            refined_query: Comprehensive problem analysis
            original_query: Original user query

        Returns:
            Workflow dict with nodes and edges following execution sequence
        """
        try:
            logger.info("=" * 100)
            logger.info(
                f"🔨 [SIMPLE WORKFLOW] Generating simple sequential workflow for {len(tools)} tools"
            )
            logger.info(f"📝 [QUERY] {refined_query[:300]}...")
            logger.info("=" * 100)

            # DEDUPLICATE TOOLS: Remove duplicate tools based on normalized name
            logger.info("🔍 [DEDUPLICATION] Removing duplicate tools...")
            unique_tools = []
            seen_tool_names = set()

            for tool in tools:
                tool_title = tool.get("Title", "").strip()

                # Normalize tool name: lowercase and remove common suffixes
                tool_name_normalized = tool_title.lower()
                tool_name_normalized = (
                    tool_name_normalized.replace(".com", "")
                    .replace(".ai", "")
                    .replace(".io", "")
                    .replace(".net", "")
                    .replace(".org", "")
                    .strip()
                )

                # Check if we've seen this normalized name before
                if tool_name_normalized in seen_tool_names:
                    logger.info(
                        f"🔄 [DUPLICATE] Skipping duplicate tool: '{tool_title}' (normalized: '{tool_name_normalized}')"
                    )
                    continue

                # Add to unique tools
                unique_tools.append(tool)
                seen_tool_names.add(tool_name_normalized)
                logger.info(
                    f"✅ [UNIQUE] Added tool: '{tool_title}' (normalized: '{tool_name_normalized}')"
                )

            logger.info(
                f"✅ [DEDUPLICATION] Kept {len(unique_tools)}/{len(tools)} unique tools"
            )
            logger.info("=" * 100)

            # SIMPLIFIED: Just create nodes and connect them sequentially - no complex analysis
            logger.info(
                "🏗️ [SIMPLE] Creating simple sequential workflow (no complex analysis)..."
            )

            # Create nodes for all unique tools
            nodes = []
            for idx, tool in enumerate(unique_tools, 1):
                node_id = str(uuid.uuid4())
                node = {
                    "id": node_id,
                    "type": "tool",
                    "position_x": 100 + (idx - 1) * 300,
                    "position_y": 100 + (idx - 1) * 50,
                    "data": {
                        "label": tool.get("Title", "Unknown Tool"),
                        "description": tool.get("Description", "")[:200],
                        "category": tool.get("Category", "AI Tool"),
                        "website": tool.get("Website", ""),
                        "tags": tool.get("Tags (Keywords)", []),
                        "features": tool.get("Features", []),
                        "source": tool.get("Source", ""),
                        "twitter": tool.get("Twitter", ""),
                        "facebook": tool.get("Facebook", ""),
                        "linkedin": tool.get("Linkedin", ""),
                        "instagram": tool.get("Instagram", ""),
                        "execution_step": idx,
                        "workflow_stage": self._get_simple_stage(idx, len(tools)),
                        "recommendation_reason": tool.get("recommendation_reason")
                        or tool.get("Description", "")[:150]
                        or f"Tool #{idx} for your workflow",
                    },
                }
                nodes.append(node)
                logger.info(f"✅ Created node #{idx}: {tool.get('Title', 'Unknown')}")

            # Create simple sequential edges (1 → 2 → 3 → 4...)
            edges = []
            for i in range(len(nodes) - 1):
                edge_id = str(uuid.uuid4())
                edge = {
                    "id": edge_id,
                    "source": nodes[i]["id"],
                    "target": nodes[i + 1]["id"],
                    "type": "default",
                }
                edges.append(edge)
                logger.info(f"✅ Created edge: {i+1} → {i+2}")

            logger.info("=" * 100)
            logger.info(
                f"✅ [SUCCESS] Simple workflow generated: {len(nodes)} nodes, {len(edges)} edges"
            )
            logger.info("=" * 100)

            return {
                "query": original_query or refined_query,
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "generation_method": "simple_sequential",
                },
            }

        except Exception as e:
            logger.error(f"❌ Error generating workflow: {e}", exc_info=True)
            # Return empty workflow on error
            return {
                "query": original_query or refined_query,
                "nodes": [],
                "edges": [],
                "metadata": {"generation_method": "error_fallback"},
            }

    async def _analyze_execution_sequence(
        self, tools: List[Dict[str, Any]], refined_query: str
    ) -> Dict[str, Any]:
        """
        Analyze tools and determine the execution sequence.

        Args:
            tools: List of tools
            refined_query: Problem analysis

        Returns:
            Dict with execution sequence and flow logic
        """
        try:
            # Prepare tools info for LLM (use enriched data)
            tools_info = []
            for i, tool in enumerate(tools, 1):
                title = tool.get("Title", "Unknown")
                # Use problem-solving analysis if available (enriched data)
                description = tool.get(
                    "problem_solving_analysis", tool.get("Description", "")
                )[:300]
                features = tool.get("Features", [])
                if isinstance(features, list):
                    features_str = ", ".join(features[:3])
                else:
                    features_str = str(features)[:100]

                tools_info.append(
                    {
                        "index": i,
                        "title": title,
                        "description": description,
                        "features": features_str,
                        "source": tool.get("Source", "Unknown"),
                    }
                )

            prompt = f"""You are a workflow architect. Analyze these tools and create a SEQUENTIAL EXECUTION PLAN.

USER'S GOAL:
{refined_query[:800]}

AVAILABLE TOOLS:
{self._format_tools_for_prompt(tools_info)}

YOUR TASK:
Create a sequential execution plan that shows HOW these tools should be connected in a logical workflow.

EXECUTION FLOW PRINCIPLES:
1. **START** → Tools that collect/receive input data
2. **PROCESS** → Tools that transform/analyze data
3. **ACTION** → Tools that perform operations/send output
4. **MONITOR** → Tools that track results
5. **CLEANUP/RETURN** → Tools that finalize or loop back

EXAMPLE EXECUTION SEQUENCE:
Problem: "Upload resume to Google Drive, parse it, send to Slack, then delete"

Sequence:
1. Google Drive (START) - Receives resume upload
2. Resume Parser (PROCESS) - Extracts text from resume
3. Slack (ACTION) - Sends parsed content as notification
4. Google Drive (CLEANUP) - Deletes the resume file

Connections:
- Google Drive → Resume Parser (pass file)
- Resume Parser → Slack (pass parsed data)
- Slack → Google Drive (trigger cleanup after notification sent)

IMPORTANT RULES:
- Each tool should appear ONCE in the sequence (unless it has multiple distinct roles)
- Connections should follow DATA FLOW (output of one tool feeds into input of another)
- Create a LOGICAL SEQUENCE, not a mesh of all-to-all connections
- Some tools may run in parallel if they don't depend on each other
- Consider WHEN each tool should execute in the workflow
- Tools that depend on output from previous tools must come AFTER them
- **NO LOOPS**: Never create backward connections (step N cannot send to step N-1 or earlier)
- **NO UNNECESSARY EDGES**: Only connect tools that actually pass data between them
- **DAG STRUCTURE**: The workflow must be a Directed Acyclic Graph (no cycles)
- **MINIMAL CONNECTIONS**: Each tool should connect to 1-3 other tools maximum (not all tools)

Return ONLY valid JSON:
{{
  "sequence": [
    {{
      "step": 1,
      "tool_index": 1,
      "tool_name": "Tool Name",
      "stage": "START|PROCESS|ACTION|MONITOR|CLEANUP",
      "purpose": "What this tool does in the workflow",
      "receives_from": [2, 3],  // Indices of tools that send data to this tool (empty for START)
      "sends_to": [4, 5]  // Indices of tools that receive data from this tool
    }}
  ],
  "parallel_groups": [
    {{
      "steps": [2, 3],  // These steps can run in parallel
      "reason": "Both process data independently"
    }}
  ],
  "workflow_explanation": "Brief explanation of the complete workflow flow"
}}

CRITICAL: Analyze the actual tool capabilities and create a REALISTIC execution sequence.
"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if result and "sequence" in result:
                logger.info(
                    f"✅ Generated execution sequence with {len(result.get('sequence', []))} steps"
                )
                return result
            else:
                logger.error("❌ Failed to parse execution sequence from LLM")
                return None

        except Exception as e:
            logger.error(f"❌ Error analyzing execution sequence: {e}", exc_info=True)
            return None

    def _format_tools_for_prompt(self, tools_info: List[Dict]) -> str:
        """Format tools for LLM prompt."""
        formatted = []
        for tool in tools_info:
            formatted.append(
                f"{tool['index']}. **{tool['title']}**\n"
                f"   Description: {tool['description']}\n"
                f"   Features: {tool['features']}"
            )
        return "\n\n".join(formatted)

    def _create_sequential_nodes(
        self, tools: List[Dict[str, Any]], sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create workflow nodes based on execution sequence.

        Args:
            tools: Original tools list
            sequence: Execution sequence from LLM

        Returns:
            List of workflow nodes with proper positioning
        """
        try:
            nodes = []
            tool_index_to_uuid = {}

            # Create nodes following the sequence
            for seq_item in sequence:
                tool_index = seq_item.get("tool_index", 0) - 1  # Convert to 0-based
                if tool_index < 0 or tool_index >= len(tools):
                    continue

                tool = tools[tool_index]
                node_id = str(uuid.uuid4())
                tool_index_to_uuid[seq_item.get("tool_index")] = node_id

                # Extract tool data
                title = tool.get("Title", "Unknown")
                description = tool.get("Description", "")
                features = tool.get("Features", [])
                if isinstance(features, str):
                    features = [f.strip() for f in features.split(",") if f.strip()]

                tags = tool.get("Tags (Keywords)", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

                # Calculate position based on step number
                step = seq_item.get("step", 1)
                stage = seq_item.get("stage", "PROCESS")

                # Position nodes in a flowing layout
                x_position = 100 + (step - 1) * 300
                y_position = 100

                # Adjust Y position based on stage for visual clarity
                stage_offsets = {
                    "START": 0,
                    "PROCESS": 50,
                    "ACTION": 100,
                    "MONITOR": 150,
                    "CLEANUP": 200,
                }
                y_position += stage_offsets.get(stage, 50)

                # Use enriched problem-solving analysis for recommendation reason
                recommendation_reason = seq_item.get("purpose")
                if not recommendation_reason:
                    # Fallback to enriched analysis or default
                    recommendation_reason = tool.get(
                        "problem_solving_analysis",
                        f"Selected for {stage} stage in the workflow sequence",
                    )

                node = {
                    "id": node_id,
                    "type": "tool",
                    "data": {
                        "label": title,
                        "description": description,
                        "features": features,
                        "tags": tags,
                        "recommendation_reason": recommendation_reason,
                        "workflow_stage": stage.lower(),
                        "execution_step": step,
                        "website": tool.get("Website", ""),
                        "twitter": tool.get("Twitter", ""),
                        "facebook": tool.get("Facebook", ""),
                        "linkedin": tool.get("LinkedIn", ""),
                        "instagram": tool.get("Instagram", ""),
                        "source": tool.get("Source", "Unknown"),
                    },
                    "position": {"x": x_position, "y": y_position},
                }

                nodes.append(node)

            # Store mapping for edge creation
            self._tool_index_to_uuid = tool_index_to_uuid

            logger.info(f"✅ Created {len(nodes)} sequential nodes")
            return nodes

        except Exception as e:
            logger.error(f"❌ Error creating sequential nodes: {e}")
            return []

    def _create_sequential_edges(
        self, nodes: List[Dict[str, Any]], sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create edges based on execution sequence.

        Args:
            nodes: List of workflow nodes
            sequence: Execution sequence

        Returns:
            List of edges following execution flow
        """
        try:
            edges = []
            tool_index_to_uuid = getattr(self, "_tool_index_to_uuid", {})

            # Create edges based on sends_to relationships
            for seq_item in sequence:
                source_tool_index = seq_item.get("tool_index")
                source_id = tool_index_to_uuid.get(source_tool_index)

                if not source_id:
                    continue

                # Get targets this tool sends to
                sends_to = seq_item.get("sends_to", [])

                for target_tool_index in sends_to:
                    target_id = tool_index_to_uuid.get(target_tool_index)

                    if target_id and source_id != target_id:
                        edge = {
                            "id": str(uuid.uuid4()),
                            "source": source_id,
                            "target": target_id,
                            "type": "default",
                            "animated": True,  # Animate to show flow
                            "label": "data flow",
                        }
                        edges.append(edge)

            logger.info(f"✅ Created {len(edges)} sequential edges")
            return edges

        except Exception as e:
            logger.error(f"❌ Error creating sequential edges: {e}")
            return []

    def _create_intelligent_edges(
        self, nodes: List[Dict[str, Any]], sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create intelligent edges that avoid unnecessary connections and loops.

        This method ensures:
        - No loops (DAG structure)
        - Only meaningful connections based on data flow
        - No redundant edges
        - Proper sequential execution order

        Args:
            nodes: List of workflow nodes
            sequence: Execution sequence

        Returns:
            List of intelligent edges
        """
        try:
            edges = []
            tool_index_to_uuid = getattr(self, "_tool_index_to_uuid", {})
            created_connections = set()  # Track created edges to avoid duplicates

            logger.info("🔗 [INTELLIGENT EDGES] Creating edges with loop prevention...")

            # Create edges based on sends_to relationships from execution plan
            for seq_item in sequence:
                source_tool_index = seq_item.get("tool_index")
                source_id = tool_index_to_uuid.get(source_tool_index)
                source_step = seq_item.get("step", 0)

                if not source_id:
                    continue

                # Get targets this tool sends to
                sends_to = seq_item.get("sends_to", [])

                for target_tool_index in sends_to:
                    target_id = tool_index_to_uuid.get(target_tool_index)

                    if not target_id or source_id == target_id:
                        continue

                    # Find target step number
                    target_step = next(
                        (
                            s.get("step", 0)
                            for s in sequence
                            if s.get("tool_index") == target_tool_index
                        ),
                        0,
                    )

                    # CRITICAL: Prevent loops - only allow forward connections
                    if target_step <= source_step:
                        logger.warning(
                            f"⚠️ [LOOP PREVENTION] Skipping backward edge: step {source_step} → step {target_step}"
                        )
                        continue

                    # Check for duplicate connections
                    connection_key = f"{source_id}->{target_id}"
                    if connection_key in created_connections:
                        logger.info(
                            f"⏭️ [DUPLICATE] Skipping duplicate edge: {connection_key}"
                        )
                        continue

                    # Create edge
                    edge = {
                        "id": str(uuid.uuid4()),
                        "source": source_id,
                        "target": target_id,
                        "type": "default",
                        "animated": True,
                        "label": "data flow",
                    }
                    edges.append(edge)
                    created_connections.add(connection_key)
                    logger.info(
                        f"✅ [EDGE] Created: step {source_step} → step {target_step}"
                    )

            logger.info(
                f"✅ [INTELLIGENT EDGES] Created {len(edges)} edges (no loops, no duplicates)"
            )
            return edges

        except Exception as e:
            logger.error(f"❌ Error creating intelligent edges: {e}")
            # Fallback to simple sequential edges
            return self._create_sequential_edges(nodes, sequence)

    def _filter_disconnected_nodes(
        self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> tuple:
        """
        Filter out nodes that have no connections (disconnected from workflow).

        Args:
            nodes: List of workflow nodes
            edges: List of workflow edges

        Returns:
            Tuple of (filtered_nodes, edges)
        """
        try:
            if not edges:
                # If no edges, keep only the first node (start node)
                logger.warning("⚠️ No edges found, keeping only first node")
                return (nodes[:1] if nodes else [], [])

            # Get all node IDs that are connected (appear in edges)
            connected_node_ids = set()
            for edge in edges:
                connected_node_ids.add(edge.get("source"))
                connected_node_ids.add(edge.get("target"))

            # Filter nodes to keep only connected ones
            filtered_nodes = []
            for node in nodes:
                node_id = node.get("id")
                if node_id in connected_node_ids:
                    filtered_nodes.append(node)
                    logger.info(
                        f"✅ [CONNECTED] Keeping node: {node.get('data', {}).get('label', 'Unknown')}"
                    )
                else:
                    logger.warning(
                        f"❌ [DISCONNECTED] Removing node: {node.get('data', {}).get('label', 'Unknown')}"
                    )

            return (filtered_nodes, edges)

        except Exception as e:
            logger.error(f"❌ Error filtering disconnected nodes: {e}")
            # Return original nodes and edges if filtering fails
            return (nodes, edges)

    def _create_fallback_sequential_workflow(
        self, tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a simple sequential workflow as fallback.

        Args:
            tools: List of tools

        Returns:
            Basic sequential workflow
        """
        try:
            logger.info("⚠️ Using fallback sequential workflow")

            nodes = []
            edges = []

            # Create nodes in simple sequence
            for i, tool in enumerate(tools):
                node_id = str(uuid.uuid4())

                title = tool.get("Title", "Unknown")
                description = tool.get("Description", "")
                features = tool.get("Features", [])
                if isinstance(features, str):
                    features = [f.strip() for f in features.split(",") if f.strip()]

                node = {
                    "id": node_id,
                    "type": "tool",
                    "data": {
                        "label": title,
                        "description": description,
                        "features": features,
                        "tags": [],
                        "recommendation_reason": f"Step {i+1} in the workflow sequence",
                        "website": tool.get("Website", ""),
                        "source": tool.get("Source", "Unknown"),
                    },
                    "position": {"x": 100 + i * 300, "y": 100},
                }

                nodes.append(node)

                # Create edge to next node
                if i > 0:
                    edge = {
                        "id": str(uuid.uuid4()),
                        "source": nodes[i - 1]["id"],
                        "target": node_id,
                        "type": "default",
                        "animated": True,
                    }
                    edges.append(edge)

            return {
                "query": "Generated workflow",
                "nodes": nodes,
                "edges": edges,
                "metadata": {"generation_method": "fallback_sequential"},
            }

        except Exception as e:
            logger.error(f"❌ Error in fallback workflow: {e}")
            return {"query": "Error", "nodes": [], "edges": []}

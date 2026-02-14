"""Workflow Update Service for modifying existing workflows."""

import logging
import time
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from workflow.utils.node_lookup import NodeLookupService, WorkflowNodeMatcher

logger = logging.getLogger(__name__)


class WorkflowUpdateService:
    """Service for updating existing workflows with new tools."""

    def __init__(self):
        """Initialize the Workflow Update service."""
        logger.info("Workflow Update service initialized")

    async def replace_tool_in_workflow(
        self,
        workflow_data: Dict[str, Any],
        node_id: str,
        new_tool_data: Dict[str, Any],
        preserve_connections: bool = True,
    ) -> Dict[str, Any]:
        """Replace a tool in an existing workflow.

        Args:
            workflow_data: Current workflow data
            node_id: ID of the node to replace
            new_tool_data: New tool data to replace with
            preserve_connections: Whether to preserve existing connections

        Returns:
            Updated workflow data
        """
        try:
            start_time = time.time()
            logger.info(f"Replacing tool in workflow: {node_id}")

            # Create a copy of the workflow to avoid modifying the original
            updated_workflow = workflow_data.copy()
            updated_workflow["nodes"] = workflow_data["nodes"].copy()
            updated_workflow["edges"] = workflow_data["edges"].copy()

            # Find and replace the node using flexible lookup (with workflow_data for id_mapping)
            target_node, match_strategy = WorkflowNodeMatcher.find_best_match(
                updated_workflow["nodes"], node_id, workflow_data=updated_workflow
            )

            if not target_node:
                # Get detailed information about available nodes for debugging
                available_nodes = NodeLookupService.get_available_nodes_info(
                    updated_workflow["nodes"]
                )

                logger.error(
                    f"Node '{node_id}' not found for replacement. Available nodes: {available_nodes}"
                )
                raise ValueError(
                    f"Node with ID '{node_id}' not found in workflow. "
                    f"Available node IDs: {[n['id'] for n in available_nodes]}"
                )

            logger.info(
                f"✅ Found node for replacement using strategy: '{match_strategy}' - Node ID: {target_node.get('id')}"
            )

            # Find the index of the node to replace
            node_found = False
            actual_node_id = target_node.get("id")

            for i, node in enumerate(updated_workflow["nodes"]):
                if node["id"] == actual_node_id:
                    # Create new node with updated tool data and ID mapping update
                    new_node = await self._create_updated_node(
                        node, new_tool_data, updated_workflow, node_id
                    )
                    updated_workflow["nodes"][i] = new_node
                    node_found = True
                    logger.info(
                        f"Replaced node {actual_node_id} (requested: {node_id}) with {new_tool_data.get('Title') or new_tool_data.get('title', 'Unknown Tool')}"
                    )
                    break

            if not node_found:
                raise ValueError(
                    f"Internal error: Found node {actual_node_id} but couldn't locate it for replacement"
                )

            # Update workflow metadata
            updated_workflow["last_modified"] = time.time()
            updated_workflow["modification_type"] = "tool_replacement"
            updated_workflow["modified_node"] = node_id

            # Optionally regenerate connections if needed
            if not preserve_connections:
                updated_workflow = await self._regenerate_workflow_connections(
                    updated_workflow
                )

            total_time = time.time() - start_time
            logger.info(f"Workflow update completed in {total_time:.2f}s")

            return {
                "status": "success",
                "workflow": updated_workflow,
                "message": f"Successfully replaced tool in node {node_id}",
                "performance": {
                    "update_time": round(total_time, 2),
                    "nodes_updated": 1,
                },
            }

        except Exception as e:
            logger.error(f"Error replacing tool in workflow: {e}")
            return {
                "status": "error",
                "message": str(e),
                "workflow": workflow_data,
            }

    async def _create_updated_node(
        self,
        original_node: Dict[str, Any],
        new_tool_data: Dict[str, Any],
        workflow_data: Dict[str, Any],
        requested_node_id: str,
    ) -> Dict[str, Any]:
        """Create an updated node with new tool data while preserving structure and swapping IDs.

        Args:
            original_node: The node being replaced
            new_tool_data: Data for the new tool
            workflow_data: The complete workflow data (to update id_mapping)
            requested_node_id: The ID used in the request (could be node ID or original_id)

        Returns:
            Updated node with swapped ID mappings
        """
        try:
            # Get the old tool's original_id
            old_original_id = original_node.get("data", {}).get("original_id")

            # Get the new tool's ID (could be in various fields)
            new_tool_id = (
                new_tool_data.get("id")
                or new_tool_data.get("ID")
                or new_tool_data.get("tool_id")
                or new_tool_data.get("original_id")
            )

            # Store the old original_id as the new tool's original_id (ID SWAP)
            # This allows the user to reference the node by the old tool's ID
            new_original_id = old_original_id if old_original_id else new_tool_id

            logger.info(
                f"🔄 ID SWAP: Node {original_node['id']} | "
                f"Old original_id: {old_original_id} → New tool ID: {new_tool_id} | "
                f"Stored as original_id: {new_original_id}"
            )

            # Preserve the original node structure
            updated_node = {
                "id": original_node["id"],  # Keep the same node ID
                "type": original_node.get("type", "tool"),
                "data": {
                    "label": new_tool_data.get("Title")
                    or new_tool_data.get("title", "Unknown Tool"),
                    "description": new_tool_data.get("Description")
                    or new_tool_data.get("description", ""),
                    "features": (
                        (
                            new_tool_data.get("Features")
                            or new_tool_data.get("features", "")
                        ).split(",")
                        if isinstance(
                            new_tool_data.get("Features")
                            or new_tool_data.get("features"),
                            str,
                        )
                        and (
                            new_tool_data.get("Features")
                            or new_tool_data.get("features")
                        )
                        else new_tool_data.get("Features")
                        or new_tool_data.get("features", [])
                        if isinstance(
                            new_tool_data.get("Features")
                            or new_tool_data.get("features"),
                            list,
                        )
                        else []
                    ),
                    "tags": (
                        (
                            new_tool_data.get("Category")
                            or new_tool_data.get("category", "")
                        ).split(",")
                        if isinstance(
                            new_tool_data.get("Category")
                            or new_tool_data.get("category"),
                            str,
                        )
                        and (
                            new_tool_data.get("Category")
                            or new_tool_data.get("category")
                        )
                        else [
                            new_tool_data.get("Category")
                            or new_tool_data.get("category", "")
                        ]
                        if (
                            new_tool_data.get("Category")
                            or new_tool_data.get("category")
                        )
                        else [new_tool_data.get("tags", "")]
                        if new_tool_data.get("tags")
                        else []
                    ),
                    "website": new_tool_data.get("Website")
                    or new_tool_data.get("website", ""),
                    "twitter": new_tool_data.get("Twitter")
                    or new_tool_data.get("twitter", ""),
                    "facebook": new_tool_data.get("Facebook")
                    or new_tool_data.get("facebook", ""),
                    "linkedin": new_tool_data.get("LinkedIn")
                    or new_tool_data.get("Linkedin")
                    or new_tool_data.get("linkedin", ""),
                    "instagram": new_tool_data.get("Instagram")
                    or new_tool_data.get("instagram", ""),
                    # Add metadata about the replacement
                    "replaced_at": time.time(),
                    "source": new_tool_data.get("Source")
                    or new_tool_data.get("source", "Unknown"),
                    "price_from": new_tool_data.get("Price From")
                    or new_tool_data.get("price_from", ""),
                    "price_to": new_tool_data.get("Price To")
                    or new_tool_data.get("price_to", ""),
                    # CRITICAL: Store the swapped original_id
                    "original_id": new_original_id,
                    "id_format": "custom",
                },
            }

            # Preserve any additional data from the original node
            if "position" in original_node:
                updated_node["position"] = original_node["position"]
            if "positionAbsolute" in original_node:
                updated_node["positionAbsolute"] = original_node["positionAbsolute"]
            if "style" in original_node:
                updated_node["style"] = original_node["style"]
            if "dragging" in original_node:
                updated_node["dragging"] = original_node["dragging"]
            if "height" in original_node:
                updated_node["height"] = original_node["height"]
            if "width" in original_node:
                updated_node["width"] = original_node["width"]
            if "selected" in original_node:
                updated_node["selected"] = original_node["selected"]

            # Update the workflow's id_mapping to reflect the new mapping
            if "id_mapping" not in workflow_data:
                workflow_data["id_mapping"] = {}

            # Add bidirectional mapping for the new original_id
            if new_original_id:
                workflow_data["id_mapping"][new_original_id] = original_node["id"]
                workflow_data["id_mapping"][original_node["id"]] = new_original_id
                logger.info(
                    f"✅ Updated id_mapping: {new_original_id} ↔ {original_node['id']}"
                )

            # Also map the new tool's ID if it's different
            if new_tool_id and new_tool_id != new_original_id:
                workflow_data["id_mapping"][new_tool_id] = original_node["id"]
                workflow_data["id_mapping"][original_node["id"]] = new_tool_id
                logger.info(
                    f"✅ Updated id_mapping: {new_tool_id} ↔ {original_node['id']}"
                )

            return updated_node

        except Exception as e:
            logger.error(f"Error creating updated node: {e}")
            # Return original node if update fails
            return original_node

    async def _regenerate_workflow_connections(
        self, workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Regenerate workflow connections using LLM if needed."""
        try:
            logger.info("Regenerating workflow connections...")

            # Extract tool information for LLM
            tools_info = []
            for node in workflow_data.get("nodes", []):
                if node.get("type") == "tool":
                    tools_info.append(
                        {
                            "id": node["id"],
                            "name": node["data"].get("label", "Unknown"),
                            "description": node["data"].get("description", ""),
                            "features": node["data"].get("features", []),
                        }
                    )

            if len(tools_info) < 2:
                return workflow_data  # No need to regenerate connections

            # Generate optimal connections using LLM
            prompt = f"""
            Given these tools in a workflow, determine the optimal connections between them:

            Tools:
            {self._format_tools_for_prompt(tools_info)}

            Original query: {workflow_data.get('query', 'Unknown task')}

            Create a logical flow that connects these tools in the most efficient sequence.
            Return only a JSON array of edges in this format:
            [
                {{"id": "edge_001", "source": "tool_001", "target": "tool_002", "type": "default"}},
                {{"id": "edge_002", "source": "tool_002", "target": "tool_003", "type": "default"}}
            ]

            Rules:
            1. Each tool should connect to the next logical tool in the sequence
            2. Start connections from trigger_start if it exists
            3. Create a linear flow unless parallel processing makes sense
            4. Use sequential edge IDs (edge_001, edge_002, etc.)
            """

            response = await get_shared_llm().generate_response(prompt)

            try:
                new_edges = await get_shared_llm().parse_json_response(response)
                if isinstance(new_edges, list) and new_edges:
                    workflow_data["edges"] = new_edges
                    logger.info(f"Regenerated {len(new_edges)} workflow connections")
                else:
                    logger.warning(
                        "Failed to parse new edges, keeping original connections"
                    )
            except Exception as e:
                logger.error(f"Error parsing new edges: {e}")

            return workflow_data

        except Exception as e:
            logger.error(f"Error regenerating workflow connections: {e}")
            return workflow_data

    def _format_tools_for_prompt(self, tools_info: List[Dict[str, Any]]) -> str:
        """Format tools information for LLM prompt."""
        formatted = []
        for tool in tools_info:
            features_str = ", ".join(tool.get("features", [])[:3])  # Limit features
            formatted.append(
                f"- {tool['id']}: {tool['name']} - {tool.get('description', '')[:100]}... (Features: {features_str})"
            )
        return "\n".join(formatted)

    async def validate_workflow_structure(
        self, workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate the structure of a workflow after updates."""
        try:
            issues = []
            warnings = []

            # Check required fields
            if "nodes" not in workflow_data:
                issues.append("Missing 'nodes' field")
            if "edges" not in workflow_data:
                issues.append("Missing 'edges' field")

            nodes = workflow_data.get("nodes", [])
            edges = workflow_data.get("edges", [])

            # Validate nodes
            node_ids = set()
            for i, node in enumerate(nodes):
                if "id" not in node:
                    issues.append(f"Node {i} missing 'id' field")
                else:
                    node_id = node["id"]
                    if node_id in node_ids:
                        issues.append(f"Duplicate node ID: {node_id}")
                    node_ids.add(node_id)

                if "type" not in node:
                    issues.append(f"Node {node.get('id', i)} missing 'type' field")

                if "data" not in node:
                    issues.append(f"Node {node.get('id', i)} missing 'data' field")

            # Validate edges
            edge_ids = set()
            for i, edge in enumerate(edges):
                if "id" not in edge:
                    issues.append(f"Edge {i} missing 'id' field")
                else:
                    edge_id = edge["id"]
                    if edge_id in edge_ids:
                        issues.append(f"Duplicate edge ID: {edge_id}")
                    edge_ids.add(edge_id)

                # Check if source and target nodes exist
                source = edge.get("source")
                target = edge.get("target")

                if source and source not in node_ids:
                    issues.append(
                        f"Edge {edge.get('id', i)} references non-existent source node: {source}"
                    )

                if target and target not in node_ids:
                    issues.append(
                        f"Edge {edge.get('id', i)} references non-existent target node: {target}"
                    )

            # Check for orphaned nodes (except trigger nodes)
            connected_nodes = set()
            for edge in edges:
                connected_nodes.add(edge.get("source"))
                connected_nodes.add(edge.get("target"))

            for node in nodes:
                node_id = node.get("id")
                if (
                    node_id not in connected_nodes
                    and node.get("type") != "trigger"
                    and node_id != "trigger_start"
                ):
                    warnings.append(
                        f"Node {node_id} is not connected to any other nodes"
                    )

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            }

        except Exception as e:
            logger.error(f"Error validating workflow structure: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
            }

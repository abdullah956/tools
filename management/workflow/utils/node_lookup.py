"""Node lookup utilities for flexible node ID handling."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class NodeLookupService:
    """Service for flexible node lookup supporting both UUID and original node IDs."""

    @staticmethod
    def is_uuid_format(node_id: str) -> bool:
        """Check if a node ID is in UUID format.

        Args:
            node_id: Node ID to check

        Returns:
            True if node_id is in UUID format, False otherwise
        """
        # UUID pattern: 8-4-4-4-12 hexadecimal digits
        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        return bool(uuid_pattern.match(node_id))

    @staticmethod
    def is_original_format(node_id: str) -> bool:
        """Check if a node ID is in original format (e.g., node_001, tool_001).

        Args:
            node_id: Node ID to check

        Returns:
            True if node_id is in original format, False otherwise
        """
        # Original format pattern: word_digits (e.g., node_001, tool_001, trigger_start)
        original_pattern = re.compile(r"^[a-zA-Z_]+[0-9]*$")
        return bool(original_pattern.match(node_id))

    @classmethod
    def find_node_by_id(
        cls,
        nodes: List[Dict[str, Any]],
        target_node_id: str,
        search_data_label: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Find a node by ID with flexible matching.

        This method supports finding nodes by:
        1. Exact ID match (UUID or original format)
        2. Original format matching against stored UUIDs via node mapping
        3. Label matching in node data (as fallback)

        Args:
            nodes: List of workflow nodes
            target_node_id: Node ID to search for (UUID or original format)
            search_data_label: Whether to search in node.data.label as fallback

        Returns:
            Matching node dictionary or None if not found
        """
        if not nodes or not target_node_id:
            logger.warning("Empty nodes list or target_node_id provided")
            return None

        target_id = target_node_id.strip()
        logger.info(f"Searching for node with ID: '{target_id}' in {len(nodes)} nodes")

        # Step 1: Try exact ID match first
        for node in nodes:
            node_id = node.get("id", "").strip()
            if node_id == target_id:
                logger.info(f"Found exact match for node ID: '{target_id}'")
                return node

        # Step 2: If target is original format, try to find by node mapping or pattern
        if cls.is_original_format(target_id):
            logger.info(
                f"Target ID '{target_id}' is in original format, searching alternatives"
            )

            # Try to find nodes with original format stored in metadata
            for node in nodes:
                node_data = node.get("data", {})

                # Check if node has original_id stored in data
                original_id = node_data.get("original_id")
                if original_id and original_id == target_id:
                    logger.info(
                        f"Found node by original_id mapping: '{target_id}' -> '{node.get('id')}'"
                    )
                    return node

                # Check if node has node_type and sequence that matches pattern
                node_type = node_data.get("node_type", node.get("type", ""))
                sequence = node_data.get("sequence")
                if node_type and sequence is not None:
                    constructed_id = f"{node_type}_{sequence:03d}"
                    if constructed_id == target_id:
                        logger.info(
                            f"Found node by pattern matching: '{target_id}' -> '{node.get('id')}'"
                        )
                        return node

        # Step 3: If target is UUID, try reverse mapping
        elif cls.is_uuid_format(target_id):
            logger.info(f"Target ID '{target_id}' is UUID format")
            # UUID matching should have been caught in step 1, but let's be thorough
            for node in nodes:
                if node.get("id") == target_id:
                    return node

        # Step 4: Fallback - search by label if enabled
        if search_data_label:
            logger.info(f"Attempting fallback search by label for: '{target_id}'")
            for node in nodes:
                node_data = node.get("data", {})
                label = node_data.get("label", "").strip()

                # Try exact label match
                if label and label.lower() == target_id.lower():
                    logger.info(
                        f"Found node by label match: '{target_id}' -> '{node.get('id')}'"
                    )
                    return node

                # Try partial label match for original format IDs
                if cls.is_original_format(target_id) and label:
                    # Extract tool name from original ID (e.g., "node_001" -> "node")
                    base_name = target_id.split("_")[0]
                    if base_name.lower() in label.lower():
                        logger.info(
                            f"Found node by partial label match: '{target_id}' -> '{node.get('id')}'"
                        )
                        return node

        logger.warning(f"No node found for ID: '{target_id}'")
        return None

    @classmethod
    def get_available_nodes_info(
        cls, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Get information about available nodes for debugging.

        Args:
            nodes: List of workflow nodes

        Returns:
            List of node info dictionaries with id, type, label, and format
        """
        available_nodes = []

        for node in nodes:
            node_id = node.get("id", "")
            node_type = node.get("type", "")
            node_data = node.get("data", {})

            # Try multiple ways to get the label
            label = (
                node_data.get("label", "")
                or node_data.get("title", "")
                or node_data.get("name", "")
                or f"Node {node_type}"
                if node_type
                else "Unknown Node"
            )

            original_id = node_data.get("original_id", "")

            # Determine ID format
            id_format = "unknown"
            if cls.is_uuid_format(node_id):
                id_format = "uuid"
            elif cls.is_original_format(node_id):
                id_format = "original"

            node_info = {
                "id": node_id,
                "type": node_type,
                "label": label,
                "id_format": id_format,
            }

            if original_id:
                node_info["original_id"] = original_id

            available_nodes.append(node_info)

        return available_nodes

    @classmethod
    def create_node_id_mapping(
        cls, original_nodes: List[Dict[str, Any]], node_id_map: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Create enhanced nodes with original ID mapping stored in data.

        This method ensures that when nodes are saved with new UUIDs,
        the original IDs are preserved in the node data for future lookup.

        Args:
            original_nodes: Original nodes with their IDs
            node_id_map: Mapping from original IDs to new UUIDs

        Returns:
            Enhanced nodes with original_id stored in data
        """
        enhanced_nodes = []

        for node in original_nodes:
            original_id = node.get("id", "")
            new_uuid = node_id_map.get(original_id)

            if new_uuid:
                # Create enhanced node with original ID preserved
                enhanced_node = node.copy()

                # Ensure data exists
                if "data" not in enhanced_node:
                    enhanced_node["data"] = {}

                # Store original ID and metadata
                enhanced_node["data"]["original_id"] = original_id
                enhanced_node["data"]["id_format"] = (
                    "original" if cls.is_original_format(original_id) else "custom"
                )

                # If original ID follows pattern, store sequence info
                if cls.is_original_format(original_id) and "_" in original_id:
                    parts = original_id.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        enhanced_node["data"]["node_type"] = "_".join(parts[:-1])
                        enhanced_node["data"]["sequence"] = int(parts[-1])

                enhanced_nodes.append(enhanced_node)
                logger.info(f"Enhanced node mapping: {original_id} -> {new_uuid}")
            else:
                # Keep original node if no mapping found
                enhanced_nodes.append(node)
                logger.warning(f"No UUID mapping found for node: {original_id}")

        return enhanced_nodes


class WorkflowNodeMatcher:
    """Advanced node matching with multiple strategies."""

    @staticmethod
    def find_best_match(
        nodes: List[Dict[str, Any]],
        target_id: str,
        strategies: Optional[List[str]] = None,
        workflow_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Find the best matching node using multiple strategies.

        Args:
            nodes: List of workflow nodes
            target_id: Target node ID to find
            strategies: List of strategies to use (default: all available)
            workflow_data: Optional workflow data containing id_mapping

        Returns:
            Tuple of (matching_node, strategy_used) or (None, "none")
        """
        if strategies is None:
            strategies = [
                "exact_id",
                "workflow_id_mapping",  # NEW: Check workflow-level id_mapping first
                "original_id_mapping",
                "pattern_matching",
                "hybrid_mapping",
                "label_exact",
                "label_partial",
            ]

        lookup_service = NodeLookupService()

        for strategy in strategies:
            result = None

            if strategy == "exact_id":
                # Direct ID match
                for node in nodes:
                    if node.get("id") == target_id:
                        result = node
                        break

            elif strategy == "workflow_id_mapping":
                # NEW: Check workflow-level id_mapping (for tool replacements)
                if workflow_data and "id_mapping" in workflow_data:
                    id_mapping = workflow_data["id_mapping"]
                    # Check if target_id is in the mapping
                    if target_id in id_mapping:
                        mapped_id = id_mapping[target_id]
                        logger.info(f"Found ID mapping: {target_id} -> {mapped_id}")
                        # Now find the node with the mapped ID
                        for node in nodes:
                            if node.get("id") == mapped_id:
                                result = node
                                break

            elif strategy == "original_id_mapping":
                # Original ID stored in node data
                for node in nodes:
                    original_id = node.get("data", {}).get("original_id")
                    if original_id == target_id:
                        result = node
                        break

            elif strategy == "pattern_matching":
                # Pattern-based matching for original format IDs
                if lookup_service.is_original_format(target_id):
                    for node in nodes:
                        node_data = node.get("data", {})
                        node_type = node_data.get("node_type")
                        sequence = node_data.get("sequence")

                        if node_type is not None and sequence is not None:
                            constructed_id = f"{node_type}_{sequence:03d}"
                            if constructed_id == target_id:
                                result = node
                                break

            elif strategy == "hybrid_mapping":
                # Enhanced hybrid mapping for cross-compatibility
                if lookup_service.is_original_format(target_id):
                    # Extract type and sequence from target_id (e.g., "node_001" -> "node", 1)
                    parts = target_id.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        target_type = "_".join(parts[:-1])
                        target_sequence = int(parts[-1])

                        # Look for nodes with matching type and sequence
                        for node in nodes:
                            node_type = node.get("type", "")
                            node_data = node.get("data", {})

                            # Check if node type matches and sequence matches
                            if node_type == target_type:
                                # Check stored sequence or calculate from position
                                stored_sequence = node_data.get("sequence")
                                if stored_sequence == target_sequence:
                                    result = node
                                    break

                            # Also check if original_id was generated and matches
                            generated_original = node_data.get("original_id")
                            if generated_original == target_id:
                                result = node
                                break

            elif strategy == "label_exact":
                # Exact label matching
                for node in nodes:
                    label = node.get("data", {}).get("label", "")
                    if label.lower() == target_id.lower():
                        result = node
                        break

            elif strategy == "label_partial":
                # Partial label matching
                for node in nodes:
                    label = node.get("data", {}).get("label", "")
                    if (
                        target_id.lower() in label.lower()
                        or label.lower() in target_id.lower()
                    ):
                        result = node
                        break

            if result:
                logger.info(f"Found node using strategy '{strategy}': {target_id}")
                return result, strategy

        logger.warning(f"No node found using any strategy for: {target_id}")
        return None, "none"

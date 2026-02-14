"""Utility functions for transforming workflow data."""

from .models import Workflow


def transform_workflow_data(input_data):
    """Transform workflow data for serialization."""
    workflow_data = input_data
    print("workflow_data", workflow_data)

    metadata = workflow_data.get("metadata", {})

    # Transform nodes
    transformed_nodes = []
    for node in workflow_data.get("nodes", []):
        print("node", node)
        transformed_node = {
            "id": node["id"],  # Keep the original ID
            "type": node["type"],
            "position": {
                "x": node["position_x"],
                "y": node["position_y"],
            },  # Extract position from data
            "positionAbsolute": {
                "x": node["position_absolute_x"],
                "y": node["position_absolute_y"],
            },  # Extract positionAbsolute from data
            "data": node["data"],
        }
        transformed_nodes.append(transformed_node)

    # Transform edges
    transformed_edges = []
    for edge in workflow_data.get("edges", []):
        transformed_edge = {
            "id": edge["id"],  # Keep the original ID
            "source": edge["source"],  # Keep the original source
            "target": edge["target"],  # Keep the original target
            "type": edge["type"],
        }
        transformed_edges.append(transformed_edge)

    workflows = Workflow.objects.filter(id=workflow_data["id"])
    metadata = workflows.first().metadata
    # Construct the final transformed workflow data
    transformed_data = {
        "nodes": transformed_nodes,
        "edges": transformed_edges,
        "metadata": metadata,  # Use metadata from input data
    }

    return transformed_data

"""Views for AI Tool Recommender app."""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List

from adrf import viewsets as async_viewsets
from asgiref.sync import sync_to_async
from django.utils import timezone
from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from workflow.utils.node_lookup import NodeLookupService, WorkflowNodeMatcher

from .ai_agents.core.background import (
    add_background_task,
    background_add_new_tools_to_pinecone,
)
from .ai_agents.explain_tool_service import generate_tool_explanation
from .ai_agents.tools.ai_tool_recommender import AIToolRecommender
from .ai_agents.tools.tool_comparison_service import ToolComparisonService
from .ai_agents.tools.workflow_update_service import WorkflowUpdateService
from .conversational_service import ConversationAI
from .models import (
    AIToolSearchLog,
    BackgroundTask,
    ChatMessage,
    ConversationSession,
    DiscoveredTool,
    RefinedQuery,
    RefineQuerySession,
    ToolComparison,
    WorkflowGeneration,
    WorkflowImplementationGuide,
)
from .query_refinement_service import QueryRefinementService
from .serializers import (
    AddWorkflowNodeSerializer,
    AIToolSearchLogSerializer,
    BackgroundTaskSerializer,
    ConversationalResponseSerializer,
    ConversationChatSerializer,
    DiscoveredToolSerializer,
    ExplainToolSerializer,
    GenerateFromChatSerializer,
    RefinedQuerySerializer,
    RefineQueryInitSerializer,
    RefineQueryResponseSerializer,
    SearchQuerySerializer,
    SearchResultSerializer,
    SelectToolSerializer,
    ToolComparisonRequestSerializer,
    ToolComparisonResultSerializer,
    ToolComparisonSerializer,
    WorkflowGenerationSerializer,
    WorkflowImplementationGuideSerializer,
    WorkflowImplementationRequestSerializer,
    WorkflowImplementationResponseSerializer,
    WorkflowToolReplaceSerializer,
    WorkflowUpdateResultSerializer,
)
from .services.workflow_implementation_service import WorkflowImplementationService
from .utils import (
    check_search_permission,
    decrement_user_search_count,
    get_user_search_quota,
)
from .workflow_generation_service import WorkflowGenerationService

logger = logging.getLogger(__name__)


async def get_workflow_with_nodes(pk):
    """Helper function to retrieve workflow data with nodes from either WorkflowGeneration or Workflow model.

    Args:
        pk: Workflow ID

    Returns:
        Tuple of (workflow_data, workflow_query, is_workflow_generation, original_workflow)
    """
    logger.info(f"🔍 Attempting to retrieve workflow with ID: {pk}")

    try:
        # First try to get WorkflowGeneration
        from .models import WorkflowGeneration

        workflow = await sync_to_async(WorkflowGeneration.objects.get)(id=pk)
        workflow_data = workflow.workflow_data
        workflow_query = workflow.query
        is_workflow_generation = True
        original_workflow = workflow

        nodes_count = len(workflow_data.get("nodes", []))
        logger.info(f"✅ Retrieved WorkflowGeneration with {nodes_count} nodes")

        # Debug: Log first few node IDs if available
        if nodes_count > 0:
            sample_nodes = workflow_data.get("nodes", [])[:3]
            sample_ids = [node.get("id", "N/A") for node in sample_nodes]
            logger.info(f"📋 Sample node IDs from WorkflowGeneration: {sample_ids}")

        # CRITICAL FIX: Also check database Node table for WorkflowGeneration workflows
        # Nodes might have been saved to database with UUIDs even if workflow_data has original format
        try:
            from workflow.models import Node

            db_nodes = await sync_to_async(list)(Node.objects.filter(workflow_id=pk))

            if db_nodes:
                logger.info(
                    f"🔍 Found {len(db_nodes)} nodes in database Node table for WorkflowGeneration"
                )

                # Create a map of existing nodes by ID for quick lookup
                existing_node_ids = {
                    node.get("id") for node in workflow_data.get("nodes", [])
                }

                # Add database nodes that aren't already in workflow_data
                for db_node in db_nodes:
                    db_node_id = str(db_node.id)

                    # If this UUID node doesn't exist in workflow_data, add it
                    if db_node_id not in existing_node_ids:
                        node_data = db_node.data or {}
                        db_workflow_node = {
                            "id": db_node_id,
                            "type": db_node.type or "tool",
                            "position": {
                                "x": db_node.position_x or 0,
                                "y": db_node.position_y or 0,
                            },
                            "positionAbsolute": {
                                "x": db_node.position_absolute_x or 0,
                                "y": db_node.position_absolute_y or 0,
                            },
                            "data": node_data,
                        }
                        workflow_data["nodes"].append(db_workflow_node)
                        logger.info(
                            f"✅ Added database node {db_node_id} to workflow_data"
                        )

                    # Also check if we need to update existing nodes with UUID mapping
                    for node in workflow_data.get("nodes", []):
                        node_data = node.get("data", {})
                        # If node has original_id that matches database node's original_id, create mapping
                        if node_data.get("original_id") and not node_data.get(
                            "database_uuid"
                        ):
                            # Check if this original_id corresponds to this database node
                            db_original_id = node_data.get("original_id")
                            if (
                                db_node.data
                                and db_node.data.get("original_id") == db_original_id
                            ):
                                # Create id_mapping entry
                                if "id_mapping" not in workflow_data:
                                    workflow_data["id_mapping"] = {}
                                workflow_data["id_mapping"][db_node_id] = node.get("id")
                                workflow_data["id_mapping"][node.get("id")] = db_node_id
                                logger.info(
                                    f"🔗 Created UUID mapping: {node.get('id')} <-> {db_node_id}"
                                )
        except Exception as db_error:
            logger.warning(
                f"⚠️ Could not check database nodes for WorkflowGeneration: {db_error}"
            )

        return workflow_data, workflow_query, is_workflow_generation, original_workflow

    except Exception as wg_error:
        logger.info(f"❌ WorkflowGeneration not found: {wg_error}")
        # If not found, try to get regular Workflow
        try:
            from workflow.models import Workflow

            workflow_obj = await sync_to_async(Workflow.objects.get)(id=pk)
            logger.info(f"✅ Found regular Workflow: {workflow_obj.name or 'Untitled'}")

            # Debug: Log metadata structure
            metadata_preview = workflow_obj.metadata or {}
            metadata_keys = list(metadata_preview.keys()) if metadata_preview else []
            logger.info(f"📋 Workflow metadata keys: {metadata_keys}")
            if "nodes" in metadata_keys:
                nodes_in_metadata = len(metadata_preview.get("nodes", []))
                logger.info(f"📊 Found {nodes_in_metadata} nodes in metadata field")

            # For regular Workflow, we need to construct workflow_data from related nodes and edges
            nodes_queryset = workflow_obj.nodes.all()
            edges_queryset = workflow_obj.edges.all()

            # Convert querysets to lists asynchronously
            nodes_list = await sync_to_async(list)(nodes_queryset)
            edges_list = await sync_to_async(list)(edges_queryset)

            logger.info(
                f"📊 Regular Workflow has {len(nodes_list)} nodes and {len(edges_list)} edges from Node/Edge tables"
            )

            # CRITICAL FIX: If no nodes in Node/Edge tables, check metadata field
            existing_metadata = workflow_obj.metadata or {}
            if len(nodes_list) == 0 and existing_metadata:
                # Check if workflow data is stored in metadata
                metadata_nodes = existing_metadata.get("nodes", [])
                metadata_edges = existing_metadata.get("edges", [])

                if len(metadata_nodes) > 0:
                    logger.info(
                        f"🔄 Found {len(metadata_nodes)} nodes in metadata field instead of Node/Edge tables"
                    )

                    # Use nodes/edges from metadata directly (they're already in workflow format)
                    workflow_data = {
                        "nodes": metadata_nodes,
                        "edges": metadata_edges,
                        "metadata": existing_metadata,
                        "id_mapping": existing_metadata.get("id_mapping", {}),
                    }

                    workflow_query = (
                        workflow_obj.user_query
                        or workflow_obj.prompt
                        or "Unknown query"
                    )
                    is_workflow_generation = False
                    original_workflow = workflow_obj

                    logger.info(
                        f"✅ Retrieved regular Workflow from metadata with {len(metadata_nodes)} nodes and {len(metadata_edges)} edges"
                    )
                    return (
                        workflow_data,
                        workflow_query,
                        is_workflow_generation,
                        original_workflow,
                    )

            # Transform nodes from model instances to workflow format with hybrid ID support
            workflow_nodes = []
            id_mapping = {}  # Create bidirectional mapping for hybrid support

            # Debug: Log sample node info from database
            if len(nodes_list) > 0:
                sample_node = nodes_list[0]
                logger.info(
                    f"📋 Sample node from DB: ID={sample_node.id[:8]}..., type={sample_node.type}, data_keys={list((sample_node.data or {}).keys())}"
                )

            for i, node in enumerate(nodes_list):
                # Create hybrid node data that supports both UUID and original format lookups
                node_data = node.data or {}

                # If original_id doesn't exist, create a reverse mapping for compatibility
                if not node_data.get("original_id"):
                    # Generate a compatible original format ID based on node type and sequence
                    node_type = node.type or "node"
                    # Use index + 1 as sequence if no sequence exists
                    sequence = i + 1

                    # Create original format ID
                    original_id = f"{node_type}_{sequence:03d}"

                    # Add the mapping to node data for future lookups
                    node_data["original_id"] = original_id
                    node_data["node_type"] = node_type
                    node_data["sequence"] = sequence
                    node_data[
                        "id_format"
                    ] = "generated"  # Mark as generated for debugging

                    logger.info(
                        f"🔄 Generated original ID mapping: {node.id} -> {original_id}"
                    )

                # Store bidirectional mapping for quick lookups
                original_id = node_data.get("original_id")
                if original_id:
                    id_mapping[original_id] = node.id  # original -> UUID
                    id_mapping[node.id] = original_id  # UUID -> original

                # Ensure node data has a label for proper node lookup
                if not node_data.get("label"):
                    # Try to get label from various sources
                    node_data["label"] = (
                        node_data.get("title", "")
                        or node_data.get("name", "")
                        or f"Tool {i+1}"
                        if node.type == "tool"
                        else f"Node {i+1}"
                    )

                workflow_node = {
                    "id": node.id,
                    "type": node.type,
                    "position": {
                        "x": node.position_x or 0,
                        "y": node.position_y or 0,
                    },
                    "positionAbsolute": {
                        "x": node.position_absolute_x or 0,
                        "y": node.position_absolute_y or 0,
                    },
                    "data": node_data,  # Use enhanced data with original ID mapping
                    "dragging": node.dragging,
                    "height": node.height,
                    "width": node.width,
                    "selected": node.selected,
                }
                workflow_nodes.append(workflow_node)

            # Log the mapping for debugging
            if len(id_mapping) > 0:
                logger.info(
                    f"🔗 Created ID mapping for {len(id_mapping) // 2} nodes: {dict(list(id_mapping.items())[:4])}..."
                )  # Show first 2 pairs
            else:
                logger.warning(
                    "⚠️  No ID mapping created - nodes may not have original_id data"
                )

            # Transform edges from model instances to workflow format
            workflow_edges = []
            for edge in edges_list:
                workflow_edge = {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "sourceHandle": edge.source_handle,
                }
                workflow_edges.append(workflow_edge)

            # Load existing id_mapping from metadata if it exists (metadata already loaded above)
            stored_id_mapping = existing_metadata.get("id_mapping", {})

            # Merge stored mappings with newly generated mappings
            # Stored mappings take precedence (they include historical replacements)
            merged_id_mapping = {**id_mapping, **stored_id_mapping}

            if stored_id_mapping:
                logger.info(
                    f"🔄 Loaded {len(stored_id_mapping)} stored ID mappings from metadata, "
                    f"merged with {len(id_mapping)} generated mappings = {len(merged_id_mapping)} total"
                )

            # Construct workflow_data from database nodes and edges
            workflow_data = {
                "nodes": workflow_nodes,
                "edges": workflow_edges,
                "metadata": existing_metadata,
                "id_mapping": merged_id_mapping,  # Use merged mapping with historical data
            }

            workflow_query = (
                workflow_obj.user_query or workflow_obj.prompt or "Unknown query"
            )
            is_workflow_generation = False
            original_workflow = workflow_obj

            logger.info(
                f"✅ Retrieved regular Workflow with {len(workflow_nodes)} nodes and {len(workflow_edges)} edges"
            )
            return (
                workflow_data,
                workflow_query,
                is_workflow_generation,
                original_workflow,
            )

        except Exception as w_error:
            logger.error(f"❌ Regular Workflow not found: {w_error}")
            raise Exception(
                f"Workflow with ID '{pk}' not found in either WorkflowGeneration or Workflow models: WorkflowGeneration error: {wg_error}, Workflow error: {w_error}"
            )


async def update_workflow_with_nodes(
    workflow_data, is_workflow_generation, original_workflow
):
    """Helper function to save workflow data back to appropriate model.

    Args:
        workflow_data: Updated workflow data
        is_workflow_generation: Whether this is a WorkflowGeneration or regular Workflow
        original_workflow: Original workflow object
    """
    if is_workflow_generation:
        # Update WorkflowGeneration
        original_workflow.workflow_data = workflow_data
        await sync_to_async(original_workflow.save)(update_fields=["workflow_data"])
        logger.info("✅ Updated WorkflowGeneration")
    else:
        # Update regular Workflow - need to save nodes and edges back to database
        # Update the metadata AND id_mapping
        metadata = workflow_data.get("metadata", {})

        # CRITICAL FIX: Include id_mapping in metadata for persistence
        if "id_mapping" in workflow_data:
            metadata["id_mapping"] = workflow_data["id_mapping"]
            logger.info(
                f"💾 Saving id_mapping with {len(workflow_data['id_mapping'])} entries to metadata"
            )

        original_workflow.metadata = metadata
        await sync_to_async(original_workflow.save)(update_fields=["metadata"])

        # Find and update the specific nodes that were modified
        updated_nodes = workflow_data.get("nodes", [])
        for updated_node in updated_nodes:
            try:
                # Get the node from database and update it
                node_obj = await sync_to_async(original_workflow.nodes.get)(
                    id=updated_node["id"]
                )

                # Update node fields
                node_obj.type = updated_node["type"]
                node_obj.data = updated_node["data"]
                node_obj.position_x = updated_node.get("position", {}).get("x", 0)
                node_obj.position_y = updated_node.get("position", {}).get("y", 0)
                node_obj.position_absolute_x = updated_node.get(
                    "positionAbsolute", {}
                ).get("x", 0)
                node_obj.position_absolute_y = updated_node.get(
                    "positionAbsolute", {}
                ).get("y", 0)
                node_obj.dragging = updated_node.get("dragging", False)
                node_obj.height = updated_node.get("height")
                node_obj.width = updated_node.get("width")
                node_obj.selected = updated_node.get("selected", False)

                await sync_to_async(node_obj.save)()
                logger.info(f"✅ Updated node {node_obj.id} in database")

            except Exception as node_error:
                logger.error(
                    f"Error updating node {updated_node.get('id')}: {node_error}"
                )
                # Continue with other nodes even if one fails

        logger.info("✅ Updated regular Workflow and nodes")


@extend_schema_view(
    list=extend_schema(
        summary="List AI tool search logs",
        description="Get a list of all AI tool search logs for the current user",
        tags=["AI Tool Recommender - Search"],
    ),
)
class AIToolSearchViewSet(async_viewsets.ReadOnlyModelViewSet):
    """ViewSet for AI tool searches with async support."""

    serializer_class = AIToolSearchLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get search logs for current user."""
        if self.request.user.is_superuser:
            return AIToolSearchLog.objects.all()
        return AIToolSearchLog.objects.filter(user=self.request.user)

    @extend_schema(
        summary="Search for AI tools and generate workflow",
        description="Search for AI tools using Pinecone vector database and internet search, then generate a workflow with the found tools",
        request=SearchQuerySerializer,
        responses={200: SearchResultSerializer},
        tags=["AI Tool Recommender - Search"],
    )
    @action(detail=False, methods=["post"], url_path="search")
    async def search_tools(self, request):
        """Search for AI tools and generate workflow (matches FastAPI behavior)."""
        serializer = SearchQuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        start_time = time.time()
        query = serializer.validated_data["query"]
        workflow_id = serializer.validated_data.get("workflow_id")

        # Check search permission (sync operation)
        can_search, permission_msg, remaining_searches = await sync_to_async(
            check_search_permission
        )(request.user)

        if not can_search:
            return Response(
                {
                    "status": "error",
                    "error": "No remaining searches",
                    "message": permission_msg,
                    "remaining_searches": remaining_searches,
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        try:
            # Create or get ConversationSession for this workflow (for linking refined query)
            # Only create session if workflow_id is provided
            conversation = None
            if workflow_id:
                try:
                    conversation = await sync_to_async(
                        lambda: ConversationSession.objects.filter(
                            user=request.user,
                            current_context__workflow_id=str(workflow_id),
                            is_active=True,
                        ).latest("updated_at")
                    )()
                    logger.info(
                        f"📍 Using existing conversation session for workflow {workflow_id}"
                    )
                except ConversationSession.DoesNotExist:
                    # Create new conversation session for this direct search
                    conversation = await sync_to_async(
                        ConversationSession.objects.create
                    )(
                        user=request.user,
                        original_query=query,
                        session_id=uuid.uuid4(),
                        current_context={"workflow_id": str(workflow_id)},
                        workflow_state="workflow_ready",  # Direct to ready state
                        chat_history=[],  # No chat history for direct search
                    )
                    logger.info(
                        f"✅ Created conversation session for direct search workflow {workflow_id}"
                    )
            else:
                logger.info(
                    "ℹ️ No workflow_id provided - skipping conversation session creation"
                )

            # Initialize recommender
            recommender = AIToolRecommender()

            # Perform search (this gets tools from Pinecone and Internet)
            search_result = await recommender.search_tools(
                query=query,
                max_results=serializer.validated_data.get("max_results", 10),
                include_pinecone=serializer.validated_data.get(
                    "include_pinecone", True
                ),
                include_internet=serializer.validated_data.get(
                    "include_internet", True
                ),
            )

            # Check if search was successful
            if search_result.get("status") == "error":
                raise Exception(search_result.get("message", "Search failed"))

            tools = search_result.get("tools", [])
            workflow = None

            # Generate workflow with the found tools
            if tools:
                try:
                    logger.info(f"Generating workflow for query: {query[:50]}...")

                    # Generate workflow without timeout - unlimited processing time per user request
                    workflow = await recommender.generate_workflow(query, tools)

                    if workflow:
                        logger.info(
                            f"Generated workflow with {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges"
                        )
                    else:
                        logger.warning("Workflow generation failed, using fallback")
                except Exception as e:
                    logger.error(f"Error generating workflow: {e}")
                    workflow = None

            # Separate internet search tools for background processing
            internet_tools = [
                tool for tool in tools if "Internet Search" in tool.get("Source", "")
            ]

            # Queue background task to add internet tools to Pinecone
            if internet_tools:
                logger.info(
                    f"🌐 Found {len(internet_tools)} internet search tools to add to Pinecone"
                )

                # Log each internet tool
                for i, tool in enumerate(internet_tools, 1):
                    title = tool.get("Title", "Unknown")
                    website = tool.get("Website", "No website")
                    logger.info(
                        f"🌐 Internet tool {i}/{len(internet_tools)}: '{title}' from {website}"
                    )

                # Add background task
                task_id = add_background_task(
                    background_add_new_tools_to_pinecone, internet_tools, query
                )

                logger.info(
                    f"🔄 Background task queued: {task_id} - {len(internet_tools)} tools will be added to Pinecone"
                )
            else:
                logger.info("ℹ️ No internet search tools found to add to Pinecone")

            response_time_ms = (time.time() - start_time) * 1000

            # Decrement user's search count (sync operation)
            success, decrement_msg = await sync_to_async(decrement_user_search_count)(
                request.user
            )

            if success:
                logger.info(f"Search count updated: {decrement_msg}")

            # Get updated remaining searches
            remaining_searches = await sync_to_async(get_user_search_quota)(
                request.user
            )

            # Log the search (async database operation)
            search_log = await sync_to_async(AIToolSearchLog.objects.create)(
                user=request.user if request.user.is_authenticated else None,
                query=query,
                refined_query=search_result.get("refined_query", query),
                max_results=serializer.validated_data.get("max_results", 10),
                include_pinecone=serializer.validated_data.get(
                    "include_pinecone", True
                ),
                include_internet=serializer.validated_data.get(
                    "include_internet", True
                ),
                pinecone_results_count=len(
                    [t for t in tools if "Pinecone" in t.get("Source", "")]
                ),
                internet_results_count=len(internet_tools),
                total_results_count=len(tools),
                response_time_ms=response_time_ms,
                cache_hit=search_result.get("cache_hit", False),
                status="success",
            )

            # 🔧 CRITICAL FIX: Save workflow to database if generated
            workflow_generation = None
            final_workflow_id = None

            if workflow:
                try:
                    if workflow_id:
                        # Update existing workflow instead of creating new WorkflowGeneration
                        from workflow.models import Workflow

                        # Update or create the workflow with the provided ID
                        workflow_obj, created = await sync_to_async(
                            Workflow.objects.update_or_create
                        )(
                            id=str(workflow_id),
                            defaults={
                                "metadata": workflow,
                                "owner": request.user
                                if request.user.is_authenticated
                                else None,
                                "user_query": query,
                                "prompt": query,
                            },
                        )
                        final_workflow_id = str(workflow_obj.id)
                        logger.info(
                            f"✅ Updated existing workflow with ID: {final_workflow_id}"
                        )

                        # CRITICAL FIX: Save nodes and edges to Node/Edge tables!
                        from .workflow_generation_service import (
                            WorkflowGenerationService,
                        )

                        workflow_service = WorkflowGenerationService()
                        await workflow_service._save_to_node_edge_tables(
                            workflow_obj, workflow
                        )

                        nodes_count = len(workflow.get("nodes", []))
                        edges_count = len(workflow.get("edges", []))
                        logger.info(
                            f"✅ Saved {nodes_count} nodes and {edges_count} edges to Node/Edge tables for workflow {final_workflow_id}"
                        )
                    else:
                        # Create new WorkflowGeneration (original behavior)
                        workflow_generation = await sync_to_async(
                            WorkflowGeneration.objects.create
                        )(
                            user=request.user
                            if request.user.is_authenticated
                            else None,
                            query=query,
                            workflow_data=workflow,
                            tools_count=len(workflow.get("nodes", []))
                            - 1,  # Subtract trigger node
                            generation_method="llm" if workflow else "fallback",
                            generation_time_ms=response_time_ms,
                            search_log=search_log,
                        )
                        final_workflow_id = str(workflow_generation.id)
                        logger.info(
                            f"✅ Created new workflow generation with ID: {final_workflow_id}"
                        )
                except Exception as e:
                    logger.error(f"❌ Error saving workflow to database: {e}")
                    # Set final_workflow_id to workflow_id if provided, so at least we return something
                    if workflow_id:
                        final_workflow_id = str(workflow_id)
            else:
                logger.warning("⚠️ No workflow generated to save")
                # If no workflow was generated but workflow_id was provided, still return it
                if workflow_id:
                    final_workflow_id = str(workflow_id)

            # Get refined query from search result
            refined_query = search_result.get("refined_query", query)

            # 💾 SAVE REFINED QUERY TO DATABASE
            if workflow_id and refined_query:
                try:
                    from .models import RefinedQuery

                    # Create or update RefinedQuery in database
                    refined_query_obj, created = await sync_to_async(
                        RefinedQuery.objects.update_or_create
                    )(
                        workflow_id=workflow_id,
                        defaults={
                            "user": request.user
                            if request.user.is_authenticated
                            else None,
                            "session": conversation,  # ✅ Link to conversation session
                            "original_query": query,
                            "refined_query": refined_query,
                            "workflow_info": {
                                "source": "direct_search",
                                "tools_found": len(tools),
                                "workflow_generated": workflow is not None,
                            },
                        },
                    )

                    action = "created" if created else "updated"
                    logger.info(
                        f"✅ Refined query {action} in database for workflow {workflow_id}"
                    )
                    logger.info(f"📝 Refined query: {refined_query[:100]}...")

                except Exception as e:
                    logger.error(
                        f"⚠️ Failed to save refined query to database: {e}",
                        exc_info=True,
                    )

            # Update conversation session with workflow data
            if workflow and conversation:
                try:
                    conversation.workflow_nodes = workflow.get("nodes", [])
                    conversation.workflow_edges = workflow.get("edges", [])
                    await sync_to_async(conversation.save)()
                    logger.info(
                        f"✅ Updated conversation session with workflow data ({len(workflow.get('nodes', []))} nodes)"
                    )
                except Exception as e:
                    logger.error(f"⚠️ Failed to update conversation session: {e}")

            # Return response matching FastAPI format
            return Response(
                {
                    "status": "success",
                    "workflow": workflow,
                    "workflow_id": final_workflow_id,
                    "query": query,
                    "refined_query": refined_query,
                    "is_refined": True,
                    "progress_percentage": 100,
                    "cached": search_result.get("cache_hit", False),
                    "message": (
                        f"Generated workflow with {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges"
                        if workflow
                        else "No workflow generated"
                    ),
                    "new_tools_discovered": len(internet_tools),
                    "auto_discovery": {
                        "enabled": True,
                        "new_tools_queued": len(internet_tools),
                    },
                    "remaining_searches": remaining_searches,
                }
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error in search_tools: {e}")

            # Log the error (async database operation)
            await sync_to_async(AIToolSearchLog.objects.create)(
                user=request.user if request.user.is_authenticated else None,
                query=query,
                max_results=serializer.validated_data.get("max_results", 10),
                response_time_ms=response_time_ms,
                status="error",
                error_message=str(e),
            )

            return Response(
                {
                    "status": "error",
                    "error": str(e),
                    "query": query,
                    "message": f"Search failed: {str(e)}",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ==================== CONVERSATIONAL AI ENDPOINTS ====================

    @extend_schema(
        summary="Conversational workflow generation with agent routing",
        description="Unified chat endpoint that routes to appropriate agent (refine_query_generator or workflow_builder) based on context and agent parameter",
        request=ConversationChatSerializer,
        responses={200: ConversationalResponseSerializer},
        tags=["AI Tool Recommender - Conversation"],
    )
    @action(detail=False, methods=["post"], url_path="chat")
    async def conversational_chat(self, request):
        """
        Handle conversational chat with agent routing.

        This endpoint routes messages to the appropriate agent:
        - refine_query_generator: Asks 10 questions to build refined query
        - workflow_builder: Builds workflows from refined queries

        Agent selection:
        1. Explicit agent parameter in request
        2. Auto-routing based on conversation state
        3. Message content analysis
        """
        start_time = time.time()
        serializer = ConversationChatSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user_message = serializer.validated_data["query"]
        workflow_id = serializer.validated_data["workflow_id"]
        agent_name = serializer.validated_data.get("agent")  # Optional agent parameter

        # Log request start
        logger.info("=" * 80)
        logger.info(f"🚀 CHAT REQUEST STARTED - Query: {user_message[:100]}...")
        logger.info("=" * 80)

        # Check search permission (sync operation)
        can_search, permission_msg, remaining_searches = await sync_to_async(
            check_search_permission
        )(request.user)

        if not can_search:
            return Response(
                {
                    "error": "Search limit exceeded",
                    "message": permission_msg,
                    "remaining_searches": remaining_searches,
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # Wrap main processing - timeout removed per user request for unlimited processing time
        async def _process_chat():
            # Get or create conversation session based on workflow_id and user
            try:
                conversation = await sync_to_async(
                    lambda: ConversationSession.objects.filter(
                        user=request.user,
                        current_context__workflow_id=str(workflow_id),
                        is_active=True,
                    ).latest("updated_at")
                )()
            except ConversationSession.DoesNotExist:
                # Create new conversation for this workflow
                conversation = await sync_to_async(ConversationSession.objects.create)(
                    user=request.user,
                    original_query=user_message,
                    session_id=uuid.uuid4(),
                    current_context={"workflow_id": str(workflow_id)},
                    workflow_state="initial",  # Start in initial state
                )

            current_state = conversation.workflow_state
            logger.info(
                f"📍 Current state: {current_state}, Agent: {agent_name or 'auto'}, Message: '{user_message}'"
            )

            # Initialize agent router
            from ai_tool_recommender.agents import AgentRouter

            agent_router = AgentRouter()

            # Get the last agent from ChatMessage objects (for tracking user responses)
            # Optimize: Only fetch agent_name field to reduce data transfer
            last_agent = None
            try:
                last_chat_message = await sync_to_async(
                    lambda: ChatMessage.objects.filter(session=conversation)
                    .only("agent_name")
                    .order_by("-created_at")
                    .first()
                )()
                if last_chat_message:
                    last_agent = last_chat_message.agent_name
            except Exception as e:
                logger.warning(f"Could not get last agent: {e}")
                last_agent = None

            # Route message to appropriate agent
            try:
                response_data = await agent_router.route_message(
                    user_message=user_message,
                    conversation=conversation,
                    workflow_id=str(workflow_id),
                    request_user=request.user,
                    agent_name=agent_name,  # Optional explicit agent
                    max_results=serializer.validated_data.get("max_results", 10),
                    include_pinecone=serializer.validated_data.get(
                        "include_pinecone", True
                    ),
                    include_internet=serializer.validated_data.get(
                        "include_internet", True
                    ),
                    tool_id=serializer.validated_data.get(
                        "tool_id"
                    ),  # Optional tool_id for tool_assistant
                    context=serializer.validated_data.get(
                        "context"
                    ),  # Optional context for implementation_chat
                )

                # Get agent name from response
                current_agent = response_data.get("agent", "unknown")

            except Exception as e:
                logger.error(f"❌ Error routing message: {e}", exc_info=True)
                response_data = {
                    "message": "I encountered an error. Please try again.",
                    "tools_mentioned": [],
                    "workflow_changes": {},
                    "suggestions": ["Try again", "Start over"],
                    "agent": "error",
                }
                current_agent = "error"

            return conversation, response_data, current_agent, last_agent

        # Execute without timeout - unlimited processing time per user request
        try:
            (
                conversation,
                response_data,
                current_agent,
                last_agent,
            ) = await _process_chat()
        except Exception as e:
            logger.error(f"❌ Error in chat processing: {e}", exc_info=True)
            return Response(
                {
                    "error": "Processing error",
                    "message": "An error occurred while processing your request. Please try again.",
                    "status": "error",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Save chat message with agent tracking
        await sync_to_async(ChatMessage.objects.create)(
            session=conversation,
            user_message=user_message,
            ai_response=response_data["message"],
            message_type=conversation.workflow_state,
            tools_mentioned=response_data.get("tools_mentioned", []),
            workflow_changes=response_data.get("workflow_changes", {}),
            intent_analysis={},
            agent_name=current_agent,  # Track which agent responded
            user_responded_to_agent=last_agent
            or "",  # Track which agent user was responding to (empty string if None)
        )

        # Update conversation context with agent tracking
        conversation.chat_history.append(
            {
                "user": user_message,
                "ai": response_data["message"],
                "agent": current_agent,  # Track which agent responded
                "user_responded_to": last_agent,  # Track which agent user was responding to
                "timestamp": timezone.now().isoformat(),
            }
        )

        # OPTIMIZATION: Run conversation save and remaining searches query in parallel
        # These operations are independent and can be done concurrently
        _, remaining_searches = await asyncio.gather(
            sync_to_async(conversation.save)(),
            sync_to_async(get_user_search_quota)(request.user),
        )

        # Count tools mentioned
        tools_mentioned = response_data.get("tools_mentioned", [])

        # Format response
        response_payload = {
            "status": "success",
            "query": conversation.original_query,
            "tools": tools_mentioned,
            "total_count": len(tools_mentioned),
            "pinecone_count": 0,
            "internet_count": len(tools_mentioned),
            "workflow": {
                "nodes": conversation.workflow_nodes,
                "edges": conversation.workflow_edges,
            },
            "workflow_id": str(workflow_id),
            "workflow_state": conversation.workflow_state,
            "can_add_nodes": conversation.workflow_state == "workflow_ready",
            "response_time_ms": int(
                (time.time() - start_time) * 1000
            ),  # Calculate actual response time
            "cached": False,
            "message": response_data["message"],
            "agent": current_agent,  # Include which agent responded
            "new_tools_discovered": len(tools_mentioned),
            "auto_discovery": {
                "enabled": True,
                "new_tools_queued": len(tools_mentioned),
            },
            "remaining_searches": remaining_searches,
        }

        # Add progress if in questioning state
        if conversation.workflow_state == "questioning":
            from .questionnaire_service import QuestionnaireService

            q_service = QuestionnaireService()
            message, count = q_service.get_progress_message(
                conversation.questionnaire_json
            )
            response_payload["questionnaire_progress"] = message
            response_payload["question_count"] = count

        # Add workflow_changes if present
        if "workflow_changes" in response_data:
            response_payload["workflow_changes"] = response_data["workflow_changes"]

        # Add examples and tool_examples if present (from questionnaire)
        if "examples" in response_data:
            response_payload["examples"] = response_data["examples"]
        if "tool_examples" in response_data:
            response_payload["tool_examples"] = response_data["tool_examples"]

        # Add progress_percentage if present (from questionnaire)
        if "progress_percentage" in response_data:
            response_payload["progress_percentage"] = response_data[
                "progress_percentage"
            ]

        # Add is_refined flag if present (Phase 3)
        if "is_refined" in response_data:
            response_payload["is_refined"] = response_data["is_refined"]

        # Add suggestions if present (from agent)
        if "suggestions" in response_data:
            response_payload["suggestions"] = response_data["suggestions"]

        # Check if a saved refined query exists in database for this workflow_id
        # This will be null initially, but will appear once the workflow is generated
        # OPTIMIZATION: Only fetch refined_query field to reduce data transfer
        try:
            from .models import RefinedQuery

            saved_refined_query = await sync_to_async(
                lambda: RefinedQuery.objects.filter(workflow_id=workflow_id)
                .only("refined_query")
                .first()
            )()

            if saved_refined_query:
                # Add the saved refined query from database
                response_payload[
                    "refined_query_from_db"
                ] = saved_refined_query.refined_query
                # ALWAYS set is_refined to True when refined_query_from_db exists
                response_payload["is_refined"] = True
                logger.info(
                    f"📋 Including saved refined query from database for workflow {workflow_id} with is_refined=True"
                )
            else:
                # No saved refined query yet - send null for refined_query_from_db and false for is_refined
                response_payload["refined_query_from_db"] = None
                response_payload["is_refined"] = False
                logger.info(
                    f"📋 No refined query in database for workflow {workflow_id}, is_refined=False"
                )
        except Exception as e:
            logger.warning(f"Could not fetch saved refined query: {e}")
            response_payload["refined_query_from_db"] = None
            response_payload["is_refined"] = False

        # Calculate and log response time
        response_time = time.time() - start_time
        response_time_ms = int(response_time * 1000)

        # Log prominently for visibility
        logger.info("=" * 80)
        logger.info(
            f"⏱️  CHAT ENDPOINT RESPONSE TIME: {response_time:.2f}s ({response_time_ms}ms)"
        )
        logger.info(
            f"   Agent: {current_agent} | State: {conversation.workflow_state} | Tools: {len(tools_mentioned)}"
        )
        logger.info("=" * 80)

        # Decrement user's search count
        success, decrement_msg = await sync_to_async(decrement_user_search_count)(
            request.user
        )

        if success:
            logger.info(f"Search count updated: {decrement_msg}")

            # Log to SearchUsageLog for dashboard display
            from management.search.models import SearchUsageLog
            from subscription.models import UserSubscription

            # Determine search type
            search_type = (
                "trial" if request.user.trial_searches >= 0 else "subscription"
            )

            # Get current subscription if using subscription searches
            current_subscription = None
            if search_type == "subscription":
                current_subscription = await sync_to_async(
                    lambda: UserSubscription.objects.filter(
                        user=request.user, status="active", end_date__gte=timezone.now()
                    ).first()
                )()

            # Create log entry
            await sync_to_async(SearchUsageLog.objects.create)(
                user=request.user,
                subscription=current_subscription,
                search_type=search_type,
                query=user_message,
                status="success",
                response_time=response_time,
            )

        response = Response(response_payload)

        return response

    @extend_schema(
        summary="Add workflow node",
        description="Add a single node to the current workflow",
        request=AddWorkflowNodeSerializer,
        responses={200: ConversationalResponseSerializer},
        tags=["AI Tool Recommender - Conversation"],
    )
    @action(detail=False, methods=["post"], url_path="add-node")
    async def add_workflow_node(self, request):
        """Add a single node to the current workflow."""
        serializer = AddWorkflowNodeSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data["session_id"]
        tool_query = serializer.validated_data["tool_query"]

        # Get conversation context
        try:
            conversation = await sync_to_async(ConversationSession.objects.get)(
                session_id=session_id
            )
        except ConversationSession.DoesNotExist:
            return Response(
                {"error": "Conversation session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Search for specific tool
        recommender = AIToolRecommender()
        search_result = await recommender.search_tools(
            query=tool_query,
            max_results=5,
            include_pinecone=True,
            include_internet=True,
        )

        if search_result.get("status") == "error":
            return Response(
                {
                    "session_id": str(conversation.session_id),
                    "message": "Could not find tools matching your query. Try being more specific.",
                    "message_type": "tool_exploration",
                    "suggestions": [
                        "Try rephrasing your tool request",
                        "Be more specific about what you need",
                        "Describe the functionality you're looking for",
                    ],
                }
            )

        tools = search_result.get("tools", [])

        # Store available tools in conversation context
        conversation.current_context["available_tools"] = tools
        await sync_to_async(conversation.save)()

        return Response(
            {
                "session_id": str(conversation.session_id),
                "message": f"Found {len(tools)} tools matching '{tool_query}'. Which one would you like to add?",
                "message_type": "tool_exploration",
                "available_tools": tools,
                "workflow_preview": {
                    "nodes": conversation.workflow_nodes,
                    "edges": conversation.workflow_edges,
                },
                "suggestions": [
                    "Tell me which tool number you'd like to add",
                    "Or describe the specific tool you want",
                    "I can explain any of these tools in more detail",
                ],
            }
        )

    @extend_schema(
        summary="Select tool for workflow",
        description="User selects a specific tool to add as a node",
        request=SelectToolSerializer,
        responses={200: ConversationalResponseSerializer},
        tags=["AI Tool Recommender - Conversation"],
    )
    @action(detail=False, methods=["post"], url_path="select-tool")
    async def select_tool_for_node(self, request):
        """User selects a specific tool to add as a node."""
        serializer = SelectToolSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data["session_id"]
        tool_index = serializer.validated_data["tool_index"]

        try:
            conversation = await sync_to_async(ConversationSession.objects.get)(
                session_id=session_id
            )
        except ConversationSession.DoesNotExist:
            return Response(
                {"error": "Conversation session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get the selected tool from conversation context
        available_tools = conversation.current_context.get("available_tools", [])
        if tool_index >= len(available_tools):
            return Response(
                {"error": "Invalid tool index"}, status=status.HTTP_400_BAD_REQUEST
            )

        selected_tool = available_tools[tool_index]

        # Generate recommendation reason for the selected tool
        tool_name = selected_tool.get("Title", "Unknown Tool")
        tool_desc = selected_tool.get("Description", "")
        current_node_count = len(conversation.workflow_nodes)

        # Create contextual recommendation reason
        recommendation_reason = self._generate_add_node_recommendation(
            tool_name, tool_desc, current_node_count, conversation.original_query
        )

        # Create node
        node_id = f"node_{len(conversation.workflow_nodes) + 1}"
        new_node = {
            "id": node_id,
            "type": "tool",
            "data": {
                "label": tool_name,
                "description": tool_desc,
                "features": selected_tool.get("Features", []),
                "tags": selected_tool.get("Tags", []),
                "recommendation_reason": recommendation_reason,
                "website": selected_tool.get("Website", ""),
                "source": selected_tool.get("Source", "Unknown"),
            },
            "position": {"x": 100 + (len(conversation.workflow_nodes) * 200), "y": 100},
        }

        # Add to workflow
        conversation.workflow_nodes.append(new_node)
        await sync_to_async(conversation.save)()

        return Response(
            {
                "session_id": str(conversation.session_id),
                "message": f"Added {selected_tool.get('Title')} to your workflow!",
                "message_type": "workflow_building",
                "added_node": new_node,
                "workflow_preview": {
                    "nodes": conversation.workflow_nodes,
                    "edges": conversation.workflow_edges,
                },
                "suggestions": [
                    "Would you like to add another tool?",
                    "Should I connect this to other nodes?",
                    "Ready to generate the complete workflow?",
                ],
            }
        )

    @extend_schema(
        summary="Generate workflow from chat",
        description="Generate complete workflow from chat history",
        request=GenerateFromChatSerializer,
        responses={200: SearchResultSerializer},
        tags=["AI Tool Recommender - Conversation"],
    )
    @action(detail=False, methods=["post"], url_path="generate-from-chat")
    async def generate_workflow_from_chat(self, request):
        """Generate complete workflow from chat history."""
        serializer = GenerateFromChatSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        session_id = serializer.validated_data["session_id"]

        try:
            conversation = await sync_to_async(ConversationSession.objects.get)(
                session_id=session_id
            )
        except ConversationSession.DoesNotExist:
            return Response(
                {"error": "Conversation session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Initialize conversational AI
        conversation_ai = ConversationAI()

        # Generate workflow based on conversation
        if conversation.workflow_nodes:
            # Use existing nodes and generate connections
            workflow = await self._generate_workflow_from_nodes(conversation)
        else:
            # Generate complete workflow from chat history
            workflow = await conversation_ai.generate_workflow_from_chat_history(
                conversation.chat_history, conversation.original_query
            )

        if not workflow:
            return Response(
                {
                    "status": "error",
                    "message": "Could not generate workflow from conversation",
                    "suggestions": [
                        "Try adding some tools first",
                        "Be more specific about your needs",
                    ],
                }
            )

        # CRITICAL FIX: Apply UUID conversion to ensure no hardcoded IDs
        from .ai_agents.tools.ai_tool_recommender import AIToolRecommender

        recommender = AIToolRecommender()
        workflow = recommender._convert_hardcoded_ids_to_uuids(workflow)
        logger.info("✅ Applied UUID conversion to workflow from chat")

        # Save workflow
        workflow_id = str(uuid.uuid4())
        from workflow.models import Workflow

        workflow_obj = await sync_to_async(Workflow.objects.create)(
            id=workflow_id,
            metadata=workflow,
            owner=request.user,
            user_query=conversation.original_query,
        )

        # CRITICAL FIX: Save nodes and edges to Node/Edge tables!
        from .workflow_generation_service import WorkflowGenerationService

        workflow_service = WorkflowGenerationService()
        await workflow_service._save_to_node_edge_tables(workflow_obj, workflow)

        nodes_count = len(workflow.get("nodes", []))
        edges_count = len(workflow.get("edges", []))
        logger.info(
            f"✅ Saved {nodes_count} nodes and {edges_count} edges to Node/Edge tables for workflow {workflow_id}"
        )

        return Response(
            {
                "status": "success",
                "workflow": workflow,
                "workflow_id": workflow_id,
                "message": f"Generated workflow with {len(workflow.get('nodes', []))} nodes based on our conversation!",
                "session_id": str(conversation.session_id),
            }
        )

    # ==================== HELPER METHODS ====================

    def _create_tool_node(
        self, tool: Dict[str, Any], sequence_num: int
    ) -> Dict[str, Any]:
        """
        Create a standardized tool node from tool data.

        Args:
            tool: Tool dictionary with Title, Description, Features, etc.
            sequence_num: Sequence number for this node (1-based)

        Returns:
            Node dictionary with id, type, data, and position
        """
        node_id = str(uuid.uuid4())

        # Extract and clean description (remove markdown formatting)
        raw_description = tool.get("Description") or tool.get("description") or ""
        clean_description = self._clean_description(raw_description)

        # Extract and normalize features (ensure it's an array)
        features = tool.get("Features") or tool.get("features") or []
        if isinstance(features, str):
            # Split comma-separated string into array
            features = [f.strip() for f in features.split(",") if f.strip()]

        # Extract and normalize tags (ensure it's an array)
        tags = tool.get("Tags (Keywords)") or tool.get("tags") or []
        if isinstance(tags, str):
            # Split comma-separated string into array
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        # Generate recommendation reason
        tool_name = tool.get("Title") or tool.get("title") or tool.get("name") or "Tool"
        recommendation_reason = self._generate_node_recommendation_simple(
            tool_name, clean_description, features, sequence_num
        )

        return {
            "id": node_id,
            "type": "tool",
            "data": {
                "label": tool_name,
                "description": clean_description,
                "features": features,
                "tags": tags,
                "recommendation_reason": recommendation_reason,
                "website": self._validate_website_url(
                    tool.get("Website") or tool.get("website") or ""
                ),
                "twitter": tool.get("Twitter") or tool.get("twitter") or "",
                "facebook": tool.get("Facebook") or tool.get("facebook") or "",
                "linkedin": tool.get("LinkedIn") or tool.get("linkedin") or "",
                "instagram": tool.get("Instagram") or tool.get("instagram") or "",
                "source": tool.get("Source") or "Pinecone Vector Database",
                "sequence": sequence_num,
            },
            "position": {
                "x": 100 + ((sequence_num - 1) % 3) * 250,  # Grid layout: 3 columns
                "y": 100 + ((sequence_num - 1) // 3) * 150,
            },
        }

    def _generate_node_recommendation_simple(
        self, tool_name: str, description: str, features: List, sequence: int
    ) -> str:
        """
        Generate a simple recommendation reason for tool nodes.

        Args:
            tool_name: Name of the tool
            description: Tool description
            features: Tool features list
            sequence: Position in workflow

        Returns:
            Recommendation reason string
        """
        desc_lower = description.lower() if description else ""

        # Detect key capabilities
        if "automate" in desc_lower:
            capability = "automation"
        elif "crm" in desc_lower or "customer" in desc_lower:
            capability = "customer relationship management"
        elif "email" in desc_lower or "marketing" in desc_lower:
            capability = "marketing and communication"
        elif "analytics" in desc_lower or "report" in desc_lower:
            capability = "analytics and reporting"
        elif "content" in desc_lower or "social" in desc_lower:
            capability = "content management"
        elif features and len(features) > 0:
            # Use first feature as capability
            first_feature = features[0] if isinstance(features, list) else str(features)
            capability = first_feature.lower()
        else:
            capability = "workflow enhancement"

        # Position-based context
        if sequence == 1:
            return f"Selected as the starting point for your workflow with strong {capability} capabilities"
        else:
            return f"Included for its {capability} features that integrate well with your workflow"

    def _validate_website_url(self, website: str) -> str:
        """
        Validate website URL - must be a valid URL or empty string.

        Args:
            website: Website string to validate

        Returns:
            Valid URL or empty string
        """
        if not website or not isinstance(website, str):
            return ""

        website = website.strip()

        # If it's a valid URL (starts with http:// or https://), return it
        if website.startswith("http://") or website.startswith("https://"):
            return website

        # If it's not a valid URL (e.g., contains tool name), return empty string
        return ""

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

    def _generate_add_node_recommendation(
        self,
        tool_name: str,
        tool_description: str,
        current_nodes: int,
        original_query: str,
    ) -> str:
        """
        Generate a recommendation reason when user manually adds a node.

        Args:
            tool_name: Name of the tool being added
            tool_description: Description of the tool
            current_nodes: Number of nodes already in workflow
            original_query: User's original workflow goal

        Returns:
            Recommendation reason string
        """
        # Extract key value from description
        desc_lower = tool_description.lower() if tool_description else ""

        # Keywords indicating specific value
        if "automate" in desc_lower or "automation" in desc_lower:
            value_prop = "automation capabilities"
        elif "manage" in desc_lower or "management" in desc_lower:
            value_prop = "management features"
        elif "track" in desc_lower or "tracking" in desc_lower:
            value_prop = "tracking and monitoring"
        elif "analyze" in desc_lower or "analytics" in desc_lower:
            value_prop = "analytics and insights"
        elif "integrate" in desc_lower or "integration" in desc_lower:
            value_prop = "integration capabilities"
        elif "collaborate" in desc_lower or "collaboration" in desc_lower:
            value_prop = "collaboration tools"
        else:
            value_prop = "essential functionality"

        # Create contextual reason based on workflow position
        if current_nodes == 0:
            return f"Added as your first tool to establish the foundation for your workflow with {value_prop}"
        else:
            return f"Manually selected to enhance your workflow with {value_prop} that complement your existing {current_nodes} tool{'s' if current_nodes > 1 else ''}"

    async def _search_tools(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for tools with standard parameters.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            Search result dictionary with status and tools
        """
        recommender = AIToolRecommender()
        return await recommender.search_tools(
            query=query,
            max_results=max_results,
            include_pinecone=True,
            include_internet=True,
        )

    async def _safe_update_workflow_incrementally(
        self, conversation: ConversationSession, user
    ):
        """
        Safely update workflow incrementally (non-critical operation).

        Args:
            conversation: Conversation session
            user: User object
        """
        try:
            workflow_id = uuid.UUID(conversation.current_context.get("workflow_id"))
            await self._update_workflow_incrementally(workflow_id, conversation, user)
        except Exception as e:
            logger.warning(
                f"⚠️ Non-critical: Failed to update workflow incrementally: {e}"
            )

    async def _update_workflow_incrementally(
        self, workflow_id: uuid.UUID, conversation: ConversationSession, request_user
    ):
        """Update or create Workflow model incrementally as nodes are added."""
        try:
            from workflow.models import Workflow

            # Create workflow data structure
            workflow_data = {
                "nodes": conversation.workflow_nodes,
                "edges": conversation.workflow_edges,
                "metadata": {
                    "incremental_update": True,
                    "last_updated": timezone.now().isoformat(),
                    "node_count": len(conversation.workflow_nodes),
                    "edge_count": len(conversation.workflow_edges),
                },
            }

            # Update or create workflow (properly wrapped for async)
            def update_or_create_workflow():
                return Workflow.objects.update_or_create(
                    id=str(workflow_id),
                    defaults={
                        "metadata": workflow_data,
                        "owner": request_user,
                        "user_query": conversation.original_query,
                        "prompt": conversation.original_query,
                        "name": f"Workflow - {conversation.original_query[:50]}",
                    },
                )

            workflow_obj, created = await sync_to_async(update_or_create_workflow)()

            action = "Created" if created else "Updated"
            logger.info(
                f"✅ {action} Workflow model incrementally: {workflow_id} with {len(conversation.workflow_nodes)} nodes"
            )

            # CRITICAL FIX: Save nodes and edges to Node/Edge tables too!
            from .workflow_generation_service import WorkflowGenerationService

            workflow_service = WorkflowGenerationService()
            await workflow_service._save_to_node_edge_tables(
                workflow_obj, workflow_data
            )
            logger.info(
                f"✅ Saved {len(conversation.workflow_nodes)} nodes and {len(conversation.workflow_edges)} edges to Node/Edge tables"
            )

        except Exception as e:
            logger.error(f"❌ Error updating workflow incrementally: {e}")
            # Don't raise - incremental updates are non-critical

    async def _handle_conversation_intent(
        self,
        user_message: str,
        intent_analysis: Dict,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
        workflow_id: uuid.UUID = None,
    ) -> Dict[str, Any]:
        """Handle different conversation intents."""
        intent = intent_analysis.get("intent", "general_chat")

        # Debug logging
        logger.info(f"Handling intent '{intent}' for message: '{user_message}'")

        if intent == "tool_inquiry":
            logger.info(f"Calling _handle_tool_inquiry for: '{user_message}'")
            return await self._handle_tool_inquiry(
                user_message, conversation, conversation_ai
            )
        elif intent == "explore_tools":
            logger.info(f"Calling _handle_tool_exploration for: '{user_message}'")
            return await self._handle_tool_exploration(
                user_message, conversation, conversation_ai
            )
        elif intent == "add_tool":
            return await self._handle_tool_addition(
                user_message, conversation, conversation_ai
            )
        elif intent == "delete_tool":
            return await self._handle_tool_deletion(
                user_message, conversation, conversation_ai
            )
        elif intent == "workflow_discussion":
            return await self._handle_workflow_discussion(
                user_message, conversation, conversation_ai
            )
        elif intent == "workflow_question":
            return await self._handle_workflow_question(
                user_message, conversation, conversation_ai
            )
        elif intent == "generate_workflow":
            return await self._handle_workflow_generation(
                user_message, conversation, conversation_ai, workflow_id
            )
        else:
            return await self._handle_general_chat(
                user_message, conversation, conversation_ai
            )

    async def _handle_tool_exploration(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """Handle tool exploration requests - adds the best non-duplicate tool as a node."""
        try:
            #  REFINE QUERY WITH CONTEXT - makes search context-aware
            from .query_refinement_service import QueryRefinementService

            refinement_service = QueryRefinementService()

            # Build context from conversation
            context_parts = []
            if conversation.original_query:
                context_parts.append(f"Original goal: {conversation.original_query}")

            # Add recent chat history for context (last 3 messages)
            if conversation.chat_history:
                recent_history = conversation.chat_history[-3:]
                for msg in recent_history:
                    if msg.get("user"):
                        context_parts.append(f"User: {msg['user']}")

            context_str = " | ".join(context_parts) if context_parts else ""

            # Refine query with context
            if context_str:
                query_with_context = f"{context_str} | Current request: {user_message}"
                logger.info(f"🔍 Query with context: {query_with_context[:150]}...")
                refinement_result = await refinement_service.refine_query(
                    query_with_context
                )
            else:
                refinement_result = await refinement_service.refine_query(user_message)

            refined_query = refinement_result.get("refined_query", user_message)

            logger.info(f"🔍 Original: '{user_message}' → Refined: '{refined_query}'")

            # Search for tools using REFINED query
            search_result = await self._search_tools(refined_query, max_results=10)

            if search_result.get("status") == "success" and search_result.get("tools"):
                tools = search_result["tools"]

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
                        # Don't fail tool exploration if submission fails

                # 🎯 ADD THE BEST TOOL AS NODE (query refinement ensures variety)
                added_nodes = await self._add_tools_as_nodes(tools, conversation)

                if not added_nodes:
                    # No tools were added (shouldn't happen often with query refinement)
                    formatted_response = await conversation_ai.format_tools_response(
                        [], user_message
                    )
                    return {
                        "message": formatted_response["message"],
                        "tools_mentioned": [],
                        "suggestions": formatted_response["suggestions"],
                    }

                # Successfully added a tool - generate LLM response
                added_tool = added_nodes[0]

                # Generate dynamic LLM response for tool addition
                dynamic_message = await self._generate_tool_addition_message(
                    conversation_ai,
                    user_message,
                    added_tool,
                    len(conversation.workflow_nodes),
                )

                return {
                    "message": dynamic_message,
                    "tools_mentioned": tools,  # Show all tools found
                    "suggestions": [
                        "Add another tool",
                        "Generate workflow",
                        "Tell me more about this tool",
                    ],
                    "workflow_changes": {
                        "nodes_added": len(added_nodes),
                        "total_nodes": len(conversation.workflow_nodes),
                    },
                }
            else:
                # No tools found
                formatted_response = await conversation_ai.format_tools_response(
                    [], user_message
                )

                return {
                    "message": formatted_response["message"],
                    "tools_mentioned": [],
                    "suggestions": formatted_response["suggestions"],
                }

        except Exception as e:
            logger.error(f"Error in tool exploration: {e}")
            formatted_response = await conversation_ai.format_tools_response(
                [], user_message
            )

            return {
                "message": formatted_response["message"],
                "tools_mentioned": [],
                "suggestions": formatted_response["suggestions"],
            }

    async def _handle_tool_inquiry(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """
        Handle tool inquiry requests - search and display tools WITHOUT adding them to workflow.
        This is for when users ask "what are the tools for X" or "show me tools for Y".

        Args:
            user_message: User's inquiry message
            conversation: Current conversation session
            conversation_ai: ConversationAI instance for formatting

        Returns:
            Response dict with tools list and message (no workflow modification)
        """
        try:
            logger.info(f"🔍 Handling tool inquiry: '{user_message}'")

            # Extract what the user is asking about from their message
            # e.g., "what are the tools for email marketing" -> "email marketing"
            search_query = self._extract_inquiry_query(user_message)

            logger.info(f"📝 Extracted search query: '{search_query}'")

            # Search for tools using the extracted query
            search_result = await self._search_tools(search_query, max_results=10)

            if search_result.get("status") == "success" and search_result.get("tools"):
                tools = search_result["tools"]
                logger.info(f"✅ Found {len(tools)} tools for inquiry")

                # Format response using the dedicated tool inquiry formatter
                # This uses a different format than tool exploration (more informative, less action-oriented)
                formatted_response = await conversation_ai.format_tool_inquiry_response(
                    tools, user_message
                )

                return {
                    "message": formatted_response["message"],
                    "tools_mentioned": formatted_response.get(
                        "tools", tools[:5]
                    ),  # Return top 5 tools
                    "suggestions": formatted_response["suggestions"],
                    "inquiry_mode": True,  # Flag to indicate this is inquiry, not exploration
                    "workflow_changes": {},  # No workflow changes for inquiry
                }
            else:
                # No tools found
                logger.warning(f"⚠️ No tools found for inquiry: '{search_query}'")
                formatted_response = await conversation_ai.format_tool_inquiry_response(
                    [], user_message
                )

                return {
                    "message": formatted_response["message"],
                    "tools_mentioned": [],
                    "suggestions": formatted_response["suggestions"],
                    "inquiry_mode": True,
                    "workflow_changes": {},
                }

        except Exception as e:
            logger.error(f"❌ Error in tool inquiry: {e}", exc_info=True)
            # Fallback response
            return {
                "message": (
                    "I understand you're asking about tools, but I encountered an error. "
                    "Could you try rephrasing your question?"
                ),
                "tools_mentioned": [],
                "suggestions": [
                    "Try asking in a different way",
                    "Search for specific tools",
                ],
                "inquiry_mode": True,
                "workflow_changes": {},
            }

    def _extract_inquiry_query(self, user_message: str) -> str:
        """
        Extract the actual search query from a tool inquiry message.

        Examples:
            "what are the tools for email marketing" -> "email marketing"
            "show me tools for CRM" -> "CRM"
            "which tools can help with automation" -> "automation"
            "tools for social media management" -> "social media management"

        Args:
            user_message: User's inquiry message

        Returns:
            Extracted search query
        """
        try:
            message_lower = user_message.lower().strip()

            # Common inquiry patterns
            patterns = [
                "what are the tools for ",
                "what tools for ",
                "which tools for ",
                "show me tools for ",
                "list tools for ",
                "tell me tools for ",
                "what tools can ",
                "which tools can ",
                "tools that can ",
                "recommend tools for ",
                "suggest tools for ",
                "tools for ",
                "show tools ",
                "list tools ",
            ]

            # Try to extract query after the pattern
            for pattern in patterns:
                if pattern in message_lower:
                    # Extract everything after the pattern
                    query = message_lower.split(pattern, 1)[1].strip()

                    # Clean up common trailing words
                    query = query.rstrip("?").strip()

                    # Handle "can help with" patterns
                    if query.startswith("help with "):
                        query = query.replace("help with ", "", 1).strip()

                    if query:
                        logger.info(
                            f"✂️ Extracted query using pattern '{pattern}': '{query}'"
                        )
                        return query

            # If no pattern matched, use the whole message (minus common question words)
            # Remove common question starters
            cleaned = message_lower
            question_starters = [
                "what are ",
                "what is ",
                "which are ",
                "which is ",
                "show me ",
                "tell me ",
            ]
            for starter in question_starters:
                if cleaned.startswith(starter):
                    cleaned = cleaned[len(starter) :].strip()
                    break

            # Remove question mark
            cleaned = cleaned.rstrip("?").strip()

            logger.info(f"✂️ Using cleaned message as query: '{cleaned}'")
            return cleaned if cleaned else user_message

        except Exception as e:
            logger.error(f"Error extracting inquiry query: {e}")
            # Fallback: return original message
            return user_message

    async def _add_tools_as_nodes(
        self, tools: List[Dict[str, Any]], conversation: ConversationSession
    ) -> List[Dict[str, Any]]:
        """
        Add the best non-duplicate tool as a node to the workflow.
        Checks title only - if duplicate, tries next tool in list.

        Args:
            tools: List of tools (will try 1st, 2nd, 3rd until finding non-duplicate title)
            conversation: Current conversation session

        Returns:
            List of node objects that were added
        """
        added_nodes = []

        try:
            if not tools:
                logger.info("No tools provided to add")
                return added_nodes

            # Get existing tool titles (lowercase for comparison)
            existing_titles = {
                node.get("data", {}).get("label", "").lower().strip()
                for node in conversation.workflow_nodes
            }

            # Try each tool until we find one that's not a duplicate
            tool_to_add = None
            for i, tool in enumerate(tools, 1):
                tool_title = (
                    tool.get("Title") or tool.get("title") or tool.get("label") or ""
                )
                tool_title_lower = tool_title.lower().strip()

                if tool_title_lower not in existing_titles:
                    # Found a non-duplicate!
                    tool_to_add = tool
                    if i > 1:
                        logger.info(
                            f"⚠️ Skipped {i - 1} duplicate(s), using tool #{i}: '{tool_title}'"
                        )
                    else:
                        logger.info(f"✅ Using tool #1: '{tool_title}'")
                    break
                else:
                    logger.info(
                        f"⚠️ Tool #{i} '{tool_title}' is duplicate by title, trying next..."
                    )

            if not tool_to_add:
                logger.warning(
                    f"⚠️ All {len(tools)} tools are duplicates by title. No new tool added."
                )
                return []

            # Add the non-duplicate tool
            sequence_num = len(conversation.workflow_nodes) + 1

            # Create node using helper method
            node_data = self._create_tool_node(tool_to_add, sequence_num)

            # Add node to conversation
            conversation.workflow_nodes.append(node_data)
            added_nodes.append(node_data)

            logger.info(
                f"✅ Added tool '{node_data['data']['label']}' as node {node_data['id']}"
            )

            # 🔗 REGENERATE ALL EDGES INTELLIGENTLY after adding new node
            if len(conversation.workflow_nodes) > 1:
                logger.info(
                    f"🔄 Regenerating edges intelligently for {len(conversation.workflow_nodes)} nodes"
                )

                # Import workflow generation service
                from .workflow_generation_service import WorkflowGenerationService

                workflow_service = WorkflowGenerationService()

                # Regenerate all edges intelligently based on tool functionality
                conversation.workflow_edges = (
                    await workflow_service.regenerate_edges_intelligently(
                        conversation.workflow_nodes, conversation.original_query
                    )
                )

                logger.info(
                    f"✅ Regenerated {len(conversation.workflow_edges)} intelligent edges"
                )
            else:
                logger.info(f"ℹ️ First node - no edges needed")

            # Save conversation with new node and intelligently regenerated edges
            await sync_to_async(conversation.save)()
            logger.info(
                f"💾 Saved 1 node + edge to conversation. Total nodes: {len(conversation.workflow_nodes)}, Total edges: {len(conversation.workflow_edges)}"
            )

            # Also update Workflow model incrementally (non-critical) using helper
            try:
                user = getattr(conversation, "user", None)
                if user:
                    await self._safe_update_workflow_incrementally(conversation, user)
                else:
                    logger.warning(
                        "⚠️ conversation.user is None, skipping workflow model update in tool addition"
                    )
            except Exception as e:
                logger.warning(
                    f"⚠️ Non-critical: Workflow table update failed in tool addition: {e}",
                    exc_info=True,
                )

            return added_nodes

        except Exception as e:
            logger.error(f"❌ Error adding tool as node: {e}", exc_info=True)
            return added_nodes  # Return what we managed to add

    async def _handle_tool_addition(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """Handle tool addition requests - tries 1st, 2nd, 3rd best until finding non-duplicate."""
        try:
            # 🔥 REFINE QUERY FIRST - makes each query unique and specific
            from .query_refinement_service import QueryRefinementService

            refinement_service = QueryRefinementService()
            refinement_result = await refinement_service.refine_query(user_message)
            refined_query = refinement_result.get("refined_query", user_message)

            logger.info(f"🔍 Original: '{user_message}' → Refined: '{refined_query}'")

            # Search for the specific tool using REFINED query
            search_result = await self._search_tools(refined_query, max_results=10)

            if search_result.get("status") == "success" and search_result.get("tools"):
                tools = search_result["tools"]

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
                        # Don't fail tool addition if submission fails

                # 🎯 ADD THE BEST TOOL (query refinement ensures variety)
                added_nodes = await self._add_tools_as_nodes(tools, conversation)

                if not added_nodes:
                    # No tools were added (rare with query refinement)
                    return {
                        "message": "Couldn't add a tool for that. Try a different search term.",
                        "tools_mentioned": [],
                        "suggestions": [
                            "Try a more specific tool name",
                            "Describe the functionality you need",
                            "Search for different tools",
                        ],
                    }

                # Successfully added a tool - generate LLM response
                added_tool = added_nodes[0]

                # Generate dynamic LLM response for tool addition
                dynamic_message = await self._generate_tool_addition_message(
                    conversation_ai,
                    user_message,
                    added_tool,
                    len(conversation.workflow_nodes),
                )

                # Find the edge that was just added (if any)
                added_edge = None
                if len(conversation.workflow_edges) > 0:
                    # The last edge should be the one we just added
                    added_edge = conversation.workflow_edges[-1]

                return {
                    "message": dynamic_message,
                    "tools_mentioned": tools,
                    "workflow_changes": {
                        "added_node": added_tool,
                        "added_edge": added_edge,  # Include the new edge
                        "total_nodes": len(conversation.workflow_nodes),
                        "total_edges": len(conversation.workflow_edges),
                    },
                    "suggestions": [
                        "Add another tool",
                        "Connect these tools",
                        "Generate complete workflow",
                    ],
                }
            else:
                return {
                    "message": "Couldn't find a tool matching that. Can you be more specific about what you want to add?",
                    "tools_mentioned": [],
                    "suggestions": [
                        "Try a more specific tool name",
                        "Describe the functionality you need",
                        "Search for tools first",
                    ],
                }

        except Exception as e:
            logger.error(
                f"❌ Error in tool addition: {e}", exc_info=True
            )  # Log full traceback
            return {
                "message": "Had trouble adding that tool. Try again with a more specific request.",
                "tools_mentioned": [],
                "suggestions": [
                    "Try again",
                    "Be more specific",
                    "Search for tools first",
                ],
            }

    async def _handle_tool_deletion(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """Handle tool deletion requests - removes node and regenerates edges."""
        try:
            # Extract tool name from message (use entities from intent analysis)
            # Simple extraction: look for quoted text or text after "delete"/"remove"
            import re

            # Try to extract quoted text first
            quoted_match = re.search(r'["\']([^"\']+)["\']', user_message)
            if quoted_match:
                tool_name_to_delete = quoted_match.group(1).strip()
            else:
                # Extract text after delete/remove keywords
                delete_pattern = r"(?:delete|remove|drop|eliminate)\s+(.+?)(?:\s+(?:from|node|tool)|$)"
                match = re.search(delete_pattern, user_message.lower())
                if match:
                    tool_name_to_delete = match.group(1).strip()
                else:
                    # Fallback: just take everything after the command
                    tool_name_to_delete = re.sub(
                        r"^(delete|remove|drop|eliminate)\s+",
                        "",
                        user_message,
                        flags=re.IGNORECASE,
                    ).strip()

            logger.info(f"🗑️ Attempting to delete tool: '{tool_name_to_delete}'")

            # Find the node by title (case-insensitive partial match)
            node_to_delete = None
            node_index = -1
            tool_name_lower = tool_name_to_delete.lower()

            for idx, node in enumerate(conversation.workflow_nodes):
                node_label = node.get("data", {}).get("label", "").lower()
                if tool_name_lower in node_label or node_label in tool_name_lower:
                    node_to_delete = node
                    node_index = idx
                    logger.info(
                        f"✅ Found node to delete: '{node.get('data', {}).get('label')}' at index {idx}"
                    )
                    break

            if not node_to_delete:
                logger.warning(f"❌ Node '{tool_name_to_delete}' not found in workflow")
                return {
                    "message": f"Couldn't find a tool named '{tool_name_to_delete}' in your workflow. Check the tool name and try again.",
                    "tools_mentioned": [],
                    "available_tools": [
                        node.get("data", {}).get("label")
                        for node in conversation.workflow_nodes
                    ],
                    "suggestions": [
                        "Check the exact tool name",
                        "List current tools in workflow",
                        "Try a different tool name",
                    ],
                }

            deleted_node_label = node_to_delete.get("data", {}).get("label")
            deleted_node_id = node_to_delete.get("id")

            # Remove the node
            conversation.workflow_nodes.pop(node_index)
            logger.info(f"🗑️ Removed node '{deleted_node_label}' from workflow")

            # Remove all edges connected to this node
            original_edge_count = len(conversation.workflow_edges)
            conversation.workflow_edges = [
                edge
                for edge in conversation.workflow_edges
                if edge.get("source") != deleted_node_id
                and edge.get("target") != deleted_node_id
            ]
            removed_edges = original_edge_count - len(conversation.workflow_edges)
            logger.info(f"🔗 Removed {removed_edges} edges connected to deleted node")

            # 🔗 REGENERATE ALL EDGES INTELLIGENTLY after removing node
            if len(conversation.workflow_nodes) > 1:
                logger.info(
                    f"🔄 Regenerating edges intelligently for {len(conversation.workflow_nodes)} remaining nodes"
                )

                # Import workflow generation service
                from .workflow_generation_service import WorkflowGenerationService

                workflow_service = WorkflowGenerationService()

                # Regenerate all edges intelligently based on tool functionality
                conversation.workflow_edges = (
                    await workflow_service.regenerate_edges_intelligently(
                        conversation.workflow_nodes, conversation.original_query
                    )
                )

                logger.info(
                    f"✅ Regenerated {len(conversation.workflow_edges)} intelligent edges"
                )
            else:
                conversation.workflow_edges = []
                logger.info("ℹ️ Only 0 or 1 nodes remaining, no edges needed")

            # Save conversation with updated nodes and edges
            await sync_to_async(conversation.save)()
            logger.info(
                f"💾 Saved updated workflow: {len(conversation.workflow_nodes)} nodes, {len(conversation.workflow_edges)} edges"
            )
            logger.info(
                f"✅ Deletion successful - node '{deleted_node_label}' removed, building success response"
            )

            # Update Workflow model incrementally (non-critical, ignore errors)
            try:
                user = getattr(conversation, "user", None)
                if user:
                    await self._safe_update_workflow_incrementally(conversation, user)
                else:
                    logger.warning(
                        "⚠️ conversation.user is None, skipping workflow model update"
                    )
            except Exception as e:
                logger.warning(
                    f"⚠️ Non-critical: Workflow table update failed: {e}", exc_info=True
                )

            # Generate dynamic deletion success message
            try:
                remaining_tools = [
                    node.get("data", {}).get("label", "Unknown")
                    for node in conversation.workflow_nodes
                ]

                message = f"Removed {deleted_node_label} from your workflow. "
                if len(conversation.workflow_nodes) > 0:
                    message += (
                        f"You now have {len(conversation.workflow_nodes)} tools. "
                    )
                    message += f"Remaining: {', '.join(remaining_tools)}"
                else:
                    message += "Your workflow is now empty."

                return {
                    "message": message,
                    "tools_mentioned": [],
                    "workflow_changes": {
                        "deleted_node": {
                            "id": deleted_node_id,
                            "label": deleted_node_label,
                        },
                        "total_nodes": len(conversation.workflow_nodes),
                        "total_edges": len(conversation.workflow_edges),
                        "removed_edges": removed_edges,
                        "regenerated_edges": len(conversation.workflow_edges),
                    },
                    "suggestions": [
                        "Add another tool",
                        "Delete another tool"
                        if len(conversation.workflow_nodes) > 0
                        else "Start adding tools",
                        "View current workflow",
                    ],
                }
            except Exception as msg_error:
                logger.error(
                    f"❌ Error building success message after deletion: {msg_error}",
                    exc_info=True,
                )
                # Deletion was successful, just return a simple success message
                return {
                    "message": f"Removed {deleted_node_label} from your workflow.",
                    "tools_mentioned": [],
                    "workflow_changes": {
                        "deleted_node": {
                            "id": deleted_node_id,
                            "label": deleted_node_label,
                        },
                        "total_nodes": len(conversation.workflow_nodes),
                        "total_edges": len(conversation.workflow_edges),
                    },
                    "suggestions": ["Add another tool", "View current workflow"],
                }

        except Exception as e:
            logger.error(f"❌ Error in tool deletion: {e}", exc_info=True)
            return {
                "message": f"Had trouble deleting that tool. Try again with the exact tool name.",
                "tools_mentioned": [],
                "suggestions": [
                    "Try again with exact tool name",
                    "List current tools",
                    "Check spelling",
                ],
            }

    async def _handle_workflow_discussion(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """Handle workflow discussion."""
        response = await conversation_ai.generate_conversational_response(
            user_message,
            "workflow_discussion",
            {
                "workflow_nodes": conversation.workflow_nodes,
                "workflow_edges": conversation.workflow_edges,
            },
        )

        return {
            "message": response,
            "suggestions": [
                "Would you like to add more tools?",
                "Should I connect these tools?",
                "Ready to generate the workflow?",
            ],
        }

    async def _handle_workflow_question(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """
        Handle questions about the workflow.
        Users can ask "what does this do?", "why is this connected?", etc.
        The LLM will answer based on workflow data.
        """
        try:
            logger.info(f"❓ Handling workflow question: '{user_message}'")

            # Check if workflow exists
            if not conversation.workflow_nodes:
                return {
                    "message": "You don't have any tools in your workflow yet. Once you build a workflow, I can answer questions about it!",
                    "tools_mentioned": [],
                    "suggestions": [
                        "Let's build a workflow",
                        "Add some tools first",
                        "What can you help me with?",
                    ],
                }

            # Use the ConversationAI's answer_workflow_question method
            answer = await conversation_ai.answer_workflow_question(
                user_question=user_message,
                workflow_nodes=conversation.workflow_nodes,
                workflow_edges=conversation.workflow_edges,
                original_query=conversation.original_query,
            )

            logger.info(f"✅ Generated answer for workflow question")

            # Generate contextual suggestions based on the workflow
            suggestions = []
            if len(conversation.workflow_nodes) > 0:
                suggestions.append("Tell me more about a specific tool")
            if len(conversation.workflow_edges) > 0:
                suggestions.append("Explain the connections")
            suggestions.append("Add another tool")

            return {
                "message": answer,
                "tools_mentioned": [],
                "suggestions": suggestions,
            }

        except Exception as e:
            logger.error(f"❌ Error handling workflow question: {e}", exc_info=True)
            return {
                "message": (
                    "I can help you understand your workflow! "
                    "Ask me about what specific tools do, why tools are connected, "
                    "or how the workflow achieves your goals."
                ),
                "tools_mentioned": [],
                "suggestions": [
                    "What does this workflow do?",
                    "Why are these tools connected?",
                    "How does this help me?",
                ],
            }

    async def _generate_tool_addition_message(
        self,
        conversation_ai: ConversationAI,
        user_message: str,
        added_tool: Dict,
        total_nodes: int,
    ) -> str:
        """Generate dynamic LLM response for tool addition."""
        try:
            tool_data = added_tool.get("data", {})
            tool_name = tool_data.get("label", "Tool")
            tool_desc = tool_data.get("description", "")
            tool_features = tool_data.get("features", "")
            tool_website = tool_data.get("website", "")
            tool_tags = tool_data.get("tags", "")

            # Create rich context for LLM
            prompt = f"""
            The user said: "{user_message}"

            We just added this tool to their workflow:
            - Name: {tool_name}
            - Description: {tool_desc[:300]}...
            - Features: {tool_features}
            - Website: {tool_website}
            - Tags: {tool_tags}

            Current workflow now has {total_nodes} tools total.

            Generate a natural, human-like response that:
            1. Simply confirms you added the tool
            2. Briefly explains what it does in 1-2 sentences
            3. How it fits with their other tools

            Be casual and conversational like texting a colleague.
            NO emojis, NO exclamation marks, NO celebrations.
            Keep it short and natural (2-3 sentences max).
            """

            # Use the correct method: generate_response
            message = await conversation_ai.llm.generate_response(prompt)

            # Add workflow progress at the end
            message += f"\n\nWorkflow progress: {total_nodes} tools added"

            return message

        except Exception as e:
            logger.error(f"Error generating tool addition message: {e}")
            # Fallback to a nice default message
            tool_name = added_tool.get("data", {}).get("label", "Tool")
            return (
                f"Added {tool_name} to your workflow. "
                f"You now have {total_nodes} tools in your workflow."
            )

    async def _generate_workflow_success_message(
        self,
        conversation_ai: ConversationAI,
        user_message: str,
        nodes_count: int,
        edges_count: int,
        workflow_nodes: List[Dict],
    ) -> str:
        """Generate a conversational success message for workflow generation."""
        try:
            # Extract tool names from workflow nodes
            tool_names = []
            for node in workflow_nodes[:5]:  # Get first 5 tools
                label = node.get("data", {}).get("label", "")
                if label:
                    tool_names.append(label)

            tools_list = ", ".join(tool_names) if tool_names else "your selected tools"
            more_tools = f" and {nodes_count - 5} more" if nodes_count > 5 else ""

            # Extract tool descriptions for richer LLM context
            tool_descriptions = []
            for node in workflow_nodes[:3]:
                label = node.get("data", {}).get("label", "")
                desc = node.get("data", {}).get("description", "")[:100]
                if label and desc:
                    tool_descriptions.append(f"- {label}: {desc}...")

            tools_desc_text = "\n".join(tool_descriptions) if tool_descriptions else ""

            # Use conversation AI to generate detailed, informative response
            prompt = f"""
            The user said: "{user_message}"

            We successfully generated a workflow with:
            - {nodes_count} tools: {tools_list}{more_tools}
            - {edges_count} connections

            Tool details:
            {tools_desc_text}

            Generate a natural, casual response that:
            1. Simply tells them their workflow is ready
            2. Mentions the main tools by name and what they do
            3. Briefly explains how they connect (e.g., "Your CRM feeds into your email tool")

            Be conversational like texting a colleague.
            NO emojis, NO exclamation marks, NO celebrations, NO excessive enthusiasm.
            Keep it informative but casual (3-4 sentences).
            """

            # Use the correct method: generate_response
            message = await conversation_ai.llm.generate_response(prompt)

            # Add workflow stats at the end
            message += (
                f"\n\nWorkflow summary: {nodes_count} tools, {edges_count} connections"
            )

            return message

        except Exception as e:
            logger.error(f"Error generating conversational message: {e}")
            # Fallback to a nice default message
            return (
                f"Your workflow is ready with {nodes_count} tools connected through {edges_count} pathways. "
                f"The tools are connected based on their functionality so they work together naturally."
            )

    async def _handle_workflow_generation(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
        workflow_id: uuid.UUID = None,
    ) -> Dict[str, Any]:
        """Handle workflow generation requests using the workflow generation service."""
        try:
            # Check if we have nodes already
            if conversation.workflow_nodes:
                # Use the workflow generation service for intelligent generation
                workflow_service = WorkflowGenerationService()

                try:
                    # Use request.user instead of conversation.user to avoid async context issues
                    user = await sync_to_async(lambda: conversation.user)()
                    generation_result = (
                        await workflow_service.generate_workflow_from_conversation(
                            conversation, user, workflow_id
                        )
                    )
                except Exception as gen_error:
                    logger.error(
                        f"Error in workflow generation service: {gen_error}",
                        exc_info=True,
                    )
                    # Even if there's an error, the workflow might be partially generated
                    # Return success if we have nodes and edges
                    if conversation.workflow_nodes and conversation.workflow_edges:
                        # Generate conversational response about the workflow
                        nodes_count = len(conversation.workflow_nodes)
                        edges_count = len(conversation.workflow_edges)

                        conversational_message = (
                            await self._generate_workflow_success_message(
                                conversation_ai,
                                user_message,
                                nodes_count,
                                edges_count,
                                conversation.workflow_nodes,
                            )
                        )

                        return {
                            "message": conversational_message,
                            "workflow_changes": {
                                "workflow_created": True,
                                "workflow_id": None,
                                "nodes_count": nodes_count,
                                "edges_count": edges_count,
                                "rejected_nodes": 0,
                            },
                            "suggestions": [
                                "Tell me more about the workflow",
                                "Can you explain how these tools connect?",
                                "What should I do next?",
                            ],
                        }
                    else:
                        raise gen_error

                if generation_result.get("status") == "success":
                    # Generate conversational response
                    nodes_count = generation_result.get("metadata", {}).get(
                        "total_nodes", 0
                    )
                    edges_count = generation_result.get("metadata", {}).get(
                        "total_edges", 0
                    )

                    conversational_message = (
                        await self._generate_workflow_success_message(
                            conversation_ai,
                            user_message,
                            nodes_count,
                            edges_count,
                            conversation.workflow_nodes,
                        )
                    )

                    return {
                        "message": conversational_message,
                        "workflow_changes": {
                            "workflow_created": True,
                            "workflow_id": generation_result.get("workflow_id"),
                            "nodes_count": nodes_count,
                            "edges_count": edges_count,
                            "rejected_nodes": generation_result.get("metadata", {}).get(
                                "rejected_nodes", 0
                            ),
                        },
                        "suggestions": [
                            "Tell me more about the workflow",
                            "Can you explain how these tools connect?",
                            "What should I do next?",
                        ],
                    }
                else:
                    return {
                        "message": generation_result.get(
                            "message", "Failed to generate workflow"
                        ),
                        "suggestions": [
                            "Try again",
                            "Add more specific tools",
                            "Start over",
                        ],
                    }
            else:
                # No nodes yet - use query refinement and workflow generation service
                logger.info(
                    "📝 No nodes found, using query refinement + workflow generation"
                )

                # Use query refinement to extract and enhance the topic
                refinement_service = QueryRefinementService()
                refinement_result = await refinement_service.refine_query(
                    conversation.original_query
                )

                refined_query = refinement_result.get(
                    "refined_query", conversation.original_query
                )
                logger.info(f"📋 Refined query: '{refined_query}'")

                # Update conversation with refined query
                conversation.refined_query = (
                    refined_query  # Save to refined_query field
                )
                await sync_to_async(conversation.save)()

                # Now use workflow generation service to create workflow from chat
                workflow_service = WorkflowGenerationService()
                user = await sync_to_async(lambda: conversation.user)()
                generation_result = (
                    await workflow_service.generate_workflow_from_conversation(
                        conversation, user, workflow_id
                    )
                )

                if generation_result.get("status") == "success":
                    # Update workflow state to workflow_ready after successful generation
                    conversation.workflow_state = "workflow_ready"
                    await sync_to_async(conversation.save)()
                    logger.info(
                        "✅ Updated workflow state to 'workflow_ready' after conversation-based generation"
                    )

                    # Generate conversational response for workflow from chat
                    nodes_count = generation_result.get("metadata", {}).get(
                        "total_nodes", 0
                    )
                    edges_count = generation_result.get("metadata", {}).get(
                        "total_edges", 0
                    )

                    # Get nodes from generation result or conversation
                    workflow_nodes = (
                        generation_result.get("workflow", {}).get("nodes", [])
                        or conversation.workflow_nodes
                    )

                    conversational_message = (
                        await self._generate_workflow_success_message(
                            conversation_ai,
                            user_message,
                            nodes_count,
                            edges_count,
                            workflow_nodes,
                        )
                    )

                    return {
                        "message": conversational_message,
                        "workflow_changes": {
                            "workflow_created": True,
                            "workflow_id": generation_result.get("workflow_id"),
                            "nodes_count": nodes_count,
                            "edges_count": edges_count,
                            "rejected_nodes": generation_result.get("metadata", {}).get(
                                "rejected_nodes", 0
                            ),
                        },
                        "suggestions": [
                            "Tell me more about the workflow",
                            "Can you explain how these tools connect?",
                            "What should I do next?",
                        ],
                    }
                else:
                    return {
                        "message": generation_result.get(
                            "message", "Failed to generate workflow"
                        ),
                        "suggestions": [
                            "Try again",
                            "Add more specific tools",
                            "Start over",
                        ],
                    }

        except Exception as e:
            logger.error(f"Error in workflow generation: {e}")
            return {
                "message": "Had trouble creating your workflow. Try again.",
                "suggestions": ["Try again", "Add more tools first", "Start over"],
            }

    async def _handle_greeting(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """
        Handle greeting messages in initial state.
        NOTE: This should rarely be called now since workflow_building intent is checked first.
        """
        try:
            # Check if the greeting contains any workflow intent that slipped through
            message_lower = user_message.lower()
            workflow_keywords = [
                "automate",
                "workflow",
                "build",
                "create",
                "tools",
                "help me",
                "i want",
                "i need",
            ]

            if any(keyword in message_lower for keyword in workflow_keywords):
                # This is actually a workflow request, not just a greeting
                # Redirect to questionnaire initialization
                logger.warning(
                    f"⚠️ Greeting handler received workflow intent: '{user_message}' - redirecting to questionnaire"
                )
                return await self._initialize_questionnaire(
                    conversation, conversation_ai
                )

            # Pure greeting - keep response brief and focused on getting to business
            prompt = f"""
            User said: "{user_message}"

            Generate a brief, professional response that:
            1. Acknowledges their greeting
            2. States you help build automated workflows
            3. Asks what they want to automate

            Be direct and professional. NO emojis, NO exclamation marks.
            Keep it very short (1-2 sentences max).
            Temperature: 0.3 for consistency.
            """

            greeting_response = await conversation_ai.llm.generate_response(prompt)

            return {
                "message": greeting_response.strip(),
                "tools_mentioned": [],
                "suggestions": [
                    "I want to automate email marketing",
                    "Help me build a workflow for social media",
                    "I need tools for project management",
                ],
            }

        except Exception as e:
            logger.error(f"Error in greeting handling: {e}")
            return {
                "message": "Hi there. I'm here to help you build automated workflows. What would you like to automate?",
                "tools_mentioned": [],
                "suggestions": [
                    "I want to automate email marketing",
                    "Help me build a workflow",
                    "Show me what you can do",
                ],
            }

    async def _handle_general_chat(
        self,
        user_message: str,
        conversation: ConversationSession,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """Handle general chat - different behavior based on state."""
        try:
            # If in initial state, just have a conversation and guide them
            if conversation.workflow_state == "initial":
                prompt = f"""
                User said: "{user_message}"

                They haven't started building a workflow yet. Generate a helpful response that:
                1. Briefly answers or acknowledges their message
                2. Mentions that you can help them build automated workflows
                3. Asks if they'd like to build a workflow for something specific

                Be casual and conversational. NO emojis, NO exclamation marks.
                Keep it short (2-3 sentences).
                """

                response = await conversation_ai.llm.generate_response(prompt)

                return {
                    "message": response.strip(),
                    "tools_mentioned": [],
                    "suggestions": [
                        "I want to build a workflow",
                        "Help me automate something",
                        "What can you do?",
                    ],
                }

            # If in workflow_ready state, check if this is a PURE greeting first
            message_lower = user_message.lower().strip()
            simple_greetings = [
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

            # Check if it's a pure greeting (no workflow-related content)
            is_pure_greeting = any(
                greeting in message_lower for greeting in simple_greetings
            )

            # Additional check: make sure there are no workflow-related keywords
            workflow_keywords = [
                "add",
                "include",
                "put",
                "delete",
                "remove",
                "drop",
                "create",
                "generate",
                "workflow",
                "tool",
                "tools",
                "software",
                "platform",
                "app",
                "automate",
                "help",
                "need",
                "want",
                "can we",
                "could we",
                "should we",
                "let's",
                "please",
                "siri",
                "gpt",
                "chatgpt",
            ]
            has_workflow_content = any(
                keyword in message_lower for keyword in workflow_keywords
            )

            if is_pure_greeting and not has_workflow_content:
                # Handle as pure greeting, not tool exploration
                prompt = f"""
                User said: "{user_message}"

                They're greeting you while their workflow is ready. Generate a brief, friendly response that:
                1. Acknowledges their greeting
                2. Mentions their workflow is ready
                3. Asks what they'd like to do next

                Be casual and friendly. NO emojis, NO exclamation marks.
                Keep it short (1-2 sentences).
                """

                response = await conversation_ai.llm.generate_response(prompt)

                return {
                    "message": response.strip(),
                    "tools_mentioned": [],
                    "suggestions": [
                        "Add more tools",
                        "Generate workflow",
                        "Tell me about a tool",
                    ],
                }

            # If in workflow_ready state, search for tools and add them
            # 🔥 REFINE QUERY WITH CONTEXT - makes search context-aware
            from .query_refinement_service import QueryRefinementService

            refinement_service = QueryRefinementService()

            # Build context from conversation
            context_parts = []
            if conversation.original_query:
                context_parts.append(f"Original goal: {conversation.original_query}")

            # Add recent chat history for context (last 3 messages)
            if conversation.chat_history:
                recent_history = conversation.chat_history[-3:]
                for msg in recent_history:
                    if msg.get("user"):
                        context_parts.append(f"User: {msg['user']}")

            context_str = " | ".join(context_parts) if context_parts else ""

            # Refine query with context
            if context_str:
                query_with_context = f"{context_str} | Current request: {user_message}"
                logger.info(f"🔍 Query with context: {query_with_context[:150]}...")
                refinement_result = await refinement_service.refine_query(
                    query_with_context
                )
            else:
                refinement_result = await refinement_service.refine_query(user_message)

            refined_query = refinement_result.get("refined_query", user_message)

            logger.info(f"🔍 Original: '{user_message}' → Refined: '{refined_query}'")

            # Search for relevant tools using REFINED query
            search_result = await self._search_tools(refined_query, max_results=10)

            if search_result.get("status") == "success" and search_result.get("tools"):
                tools = search_result["tools"]

                # 🎯 ADD THE BEST TOOL AS NODE (query refinement ensures variety)
                added_nodes = await self._add_tools_as_nodes(tools, conversation)

                if not added_nodes:
                    # No tools were added (shouldn't happen often with query refinement)
                    formatted_response = await conversation_ai.format_tools_response(
                        [], user_message
                    )
                    return {
                        "message": formatted_response["message"],
                        "tools_mentioned": [],
                        "suggestions": formatted_response["suggestions"],
                    }

                # Successfully added a tool - generate LLM response
                added_tool = added_nodes[0]

                # Generate dynamic LLM response for tool addition
                dynamic_message = await self._generate_tool_addition_message(
                    conversation_ai,
                    user_message,
                    added_tool,
                    len(conversation.workflow_nodes),
                )

                return {
                    "message": dynamic_message,
                    "tools_mentioned": tools,  # Show all tools found
                    "suggestions": [
                        "Add another tool",
                        "Generate workflow",
                        "Tell me more about this tool",
                    ],
                    "workflow_changes": {
                        "nodes_added": len(added_nodes),
                        "total_nodes": len(conversation.workflow_nodes),
                    },
                }
            else:
                # No tools found
                formatted_response = await conversation_ai.format_tools_response(
                    [], user_message
                )

                return {
                    "message": formatted_response["message"],
                    "tools_mentioned": [],
                    "suggestions": formatted_response["suggestions"],
                }

        except Exception as e:
            logger.error(f"Error in general chat handling: {e}")
            formatted_response = await conversation_ai.format_tools_response(
                [], user_message
            )

            return {
                "message": formatted_response["message"],
                "tools_mentioned": [],
                "suggestions": formatted_response["suggestions"],
            }

    async def _initialize_questionnaire(
        self,
        conversation: ConversationSession,
        user_message: str,
        conversation_ai: ConversationAI,
    ) -> Dict[str, Any]:
        """
        Initialize 4-phase questionnaire for a new conversation.
        Starts with Phase 1: Intent Identification.

        NEW: Also triggers background tool search to pre-fetch relevant tools.
        """
        try:
            logger.info(f"🆕 Initializing 4-phase questionnaire for: '{user_message}'")

            # Generate Phase 1 questions using questionnaire service
            questionnaire_json = (
                await conversation_ai.questionnaire_service.initialize_questionnaire(
                    user_message
                )
            )

            # Save to conversation
            conversation.questionnaire_json = questionnaire_json
            conversation.workflow_state = "questioning"
            await sync_to_async(conversation.save)()

            # Get first question from Phase 1
            phase_1_data = questionnaire_json.get("phase_1", {})
            first_question = (
                phase_1_data.get("questions", [])[0]
                if phase_1_data.get("questions")
                else None
            )

            if not first_question:
                raise ValueError("No questions generated for Phase 1")

            # Generate intro message
            intro_message = (
                f"Great! Let me ask you a few questions to build the perfect workflow for you.\n\n"
                f"{first_question['question']}"
            )

            logger.info(
                f"✅ Questionnaire initialized - Starting Phase 1: Intent Identification"
            )

            # Get progress percentage (should be 0% at start)
            progress_percentage = questionnaire_json.get("progress_percentage", 0)

            return {
                "message": intro_message,
                "tools_mentioned": [],
                "suggestions": [],
                "examples": first_question.get("examples", []),
                "tool_examples": first_question.get("tool_examples", []),
                "progress_percentage": progress_percentage,
            }

        except Exception as e:
            logger.error(f"❌ Error initializing questionnaire: {e}", exc_info=True)
            return {
                "message": "Having trouble setting up the questionnaire. Let's try again.",
                "tools_mentioned": [],
                "suggestions": ["Describe what workflow you want to build"],
            }

    async def _handle_questionnaire_answer(
        self,
        conversation: ConversationSession,
        user_message: str,
        conversation_ai: ConversationAI,
        workflow_id: uuid.UUID,
        request_user,
    ) -> Dict[str, Any]:
        """
        Handle user's answer to current question in 4-phase system.
        Handle phase transitions and auto-generate workflow when all phases complete.
        """
        try:
            logger.info(f"💬 Processing answer: '{user_message}'")

            # Process the answer - now returns (questionnaire_json, phase_complete, all_complete)
            (
                questionnaire_json,
                phase_complete,
                all_complete,
            ) = await conversation_ai.questionnaire_service.process_answer(
                conversation.questionnaire_json, user_message
            )

            # Update conversation
            conversation.questionnaire_json = questionnaire_json
            await sync_to_async(conversation.save)()

            # 💾 SAVE UPDATED REFINED QUERY TO DATABASE (if in Phase 4 and refined query changed)
            current_phase = questionnaire_json.get("current_phase", 1)
            if current_phase == 4:
                workflow_info = questionnaire_json.get("workflow_info", {})
                refined_query = workflow_info.get("refined_query")

                if refined_query:
                    try:
                        await conversation_ai.questionnaire_service.save_refined_query_to_db(
                            refined_query=refined_query,
                            workflow_id=str(workflow_id),
                            conversation_session=conversation,
                            request_user=request_user,
                            questionnaire_json=questionnaire_json,
                        )
                        logger.info(
                            f"✅ Updated refined query saved to database after user feedback"
                        )
                    except Exception as save_error:
                        logger.error(
                            f"⚠️ Failed to save updated refined query: {save_error}"
                        )

            # Check if user needs clarification (phase_complete=False, all_complete=False)
            if not phase_complete and not all_complete:
                # User needs clarification - get current question and provide explanation
                current_phase = questionnaire_json.get("current_phase", 1)
                phase_key = f"phase_{current_phase}"
                phase_data = questionnaire_json.get(phase_key, {})
                current_index = phase_data.get("current_question_index", 0)
                questions = phase_data.get("questions", [])

                if current_index < len(questions):
                    current_question = questions[current_index]

                    # Get clarification analysis
                    clarification_analysis = await conversation_ai.questionnaire_service._analyze_user_response(
                        user_message, current_question
                    )

                    if clarification_analysis["needs_clarification"]:
                        # Provide explanation and repeat question
                        explanation = clarification_analysis.get("explanation", "")
                        if explanation:
                            message = f"{explanation}\n\n{current_question['question']}"
                        else:
                            message = current_question["question"]

                        logger.info(
                            f"🤔 Providing clarification for question: {current_question['id']}"
                        )

                        # Get progress percentage and refined query
                        progress_percentage = questionnaire_json.get(
                            "progress_percentage", 0
                        )
                        workflow_info = questionnaire_json.get("workflow_info", {})
                        refined_query_from_info = workflow_info.get("refined_query")

                        response_data = {
                            "message": message,
                            "tools_mentioned": [],
                            "suggestions": [],
                            "examples": current_question.get("examples", []),
                            "tool_examples": current_question.get("tool_examples", []),
                            "progress_percentage": progress_percentage,
                        }

                        # Add refined query if it exists
                        if refined_query_from_info:
                            response_data["refined_query"] = refined_query_from_info

                        # Add refined query fields for Phase 4 clarification (Refinement)
                        if current_phase == 4 and current_question.get("is_refined"):
                            response_data["is_refined"] = True
                            response_data["refined_query"] = current_question.get(
                                "refined_query", ""
                            )
                            logger.info(
                                f"🎯 Phase 4 clarification: Including refined query with is_refined flag"
                            )

                        return response_data

            if all_complete:
                # ALL 4 PHASES COMPLETE - AUTO-GENERATE WORKFLOW
                logger.info("🎉🎉🎉 All 4 phases complete! Auto-generating workflow...")

                # Change state to generating
                conversation.workflow_state = "generating_workflow"
                await sync_to_async(conversation.save)()

                # Generate workflow from questionnaire
                from .workflow_generation_service import WorkflowGenerationService

                workflow_service = WorkflowGenerationService()

                workflow_result = (
                    await workflow_service.generate_workflow_from_questionnaire(
                        conversation, request_user, workflow_id
                    )
                )

                if workflow_result.get("status") == "success":
                    # Change state to workflow_ready
                    conversation.workflow_state = "workflow_ready"
                    await sync_to_async(conversation.save)()

                    logger.info("✅ Workflow generated successfully!")

                    return {
                        "message": workflow_result["message"],
                        "tools_mentioned": [],
                        "workflow_changes": {
                            "nodes_added": len(
                                workflow_result["workflow"].get("nodes", [])
                            ),
                            "edges_added": len(
                                workflow_result["workflow"].get("edges", [])
                            ),
                            "generation_method": "questionnaire",
                        },
                        "suggestions": [
                            "Add more tools",
                            "Edit connections",
                            "Tell me about a tool",
                        ],
                    }
                else:
                    # Error generating workflow
                    conversation.workflow_state = "error"
                    await sync_to_async(conversation.save)()

                    return {
                        "message": workflow_result.get(
                            "message", "Failed to generate workflow"
                        ),
                        "tools_mentioned": [],
                        "suggestions": [
                            "Try starting over",
                            "Describe your needs differently",
                        ],
                    }

            elif phase_complete:
                # PHASE COMPLETE - TRANSITION TO NEXT PHASE
                current_phase = questionnaire_json.get("current_phase", 1)
                logger.info(
                    f"🎉 Phase {current_phase} complete! Transitioning to next phase..."
                )

                # Transition to next phase
                questionnaire_json = await conversation_ai.questionnaire_service.transition_to_next_phase(
                    questionnaire_json
                )

                # Update conversation
                conversation.questionnaire_json = questionnaire_json
                await sync_to_async(conversation.save)()

                # Get first question of new phase
                next_question = conversation_ai.questionnaire_service.get_next_question(
                    questionnaire_json
                )

                if next_question:
                    new_phase = questionnaire_json.get("current_phase", 1)
                    phase_key = f"phase_{new_phase}"
                    phase_data = questionnaire_json.get(phase_key, {})
                    phase_name = phase_data.get("name", f"Phase {new_phase}")

                    # For Phase 4 (Refinement), include the refined query in the message
                    if new_phase == 4:
                        refined_query = phase_data.get("refined_query", "")

                        # 💾 SAVE REFINED QUERY TO DATABASE IMMEDIATELY
                        if refined_query:
                            try:
                                await conversation_ai.questionnaire_service.save_refined_query_to_db(
                                    refined_query=refined_query,
                                    workflow_id=str(workflow_id),
                                    conversation_session=conversation,
                                    request_user=request_user,
                                    questionnaire_json=questionnaire_json,
                                )
                                logger.info(
                                    f"✅ Refined query saved to database for workflow {workflow_id}"
                                )
                            except Exception as save_error:
                                logger.error(
                                    f"⚠️ Failed to save refined query to DB: {save_error}"
                                )

                        message = (
                            f"Great! Now let's refine your workflow.\n\n"
                            f"Based on our conversation, here's what I understand:\n"
                            f"**{refined_query}**\n\n"
                            f"{next_question['question']}"
                        )
                    else:
                        message = (
                            f"Excellent! Moving to {phase_name}.\n\n"
                            f"{next_question['question']}"
                        )

                    logger.info(f"✅ Transitioned to Phase {new_phase}: {phase_name}")

                    # Get progress percentage and refined query
                    progress_percentage = questionnaire_json.get(
                        "progress_percentage", 0
                    )
                    workflow_info = questionnaire_json.get("workflow_info", {})
                    refined_query_from_info = workflow_info.get("refined_query")

                    response_data = {
                        "message": message,
                        "tools_mentioned": [],
                        "suggestions": [],
                        "examples": next_question.get("examples", []),
                        "tool_examples": next_question.get("tool_examples", []),
                        "progress_percentage": progress_percentage,
                    }

                    # Add refined query if it exists
                    if refined_query_from_info:
                        response_data["refined_query"] = refined_query_from_info

                    # Add is_refined flag for Phase 4 (Refinement)
                    if new_phase == 4 and next_question.get("is_refined"):
                        response_data["is_refined"] = True

                    return response_data
                else:
                    # Should not reach here
                    return {
                        "message": "Something went wrong with phase transition. Let's start over.",
                        "tools_mentioned": [],
                        "suggestions": ["Start a new workflow"],
                    }

            else:
                # More questions in current phase
                next_question = conversation_ai.questionnaire_service.get_next_question(
                    questionnaire_json
                )

                if next_question:
                    message = next_question["question"]

                    current_phase = questionnaire_json.get("current_phase", 1)
                    phase_key = f"phase_{current_phase}"
                    phase_data = questionnaire_json.get(phase_key, {})
                    current_index = phase_data.get("current_question_index", 0)
                    total_phase_questions = len(phase_data.get("questions", []))

                    logger.info(
                        f"📋 Phase {current_phase}: Asking question {current_index + 1}/{total_phase_questions}"
                    )

                    # Get progress percentage and refined query
                    progress_percentage = questionnaire_json.get(
                        "progress_percentage", 0
                    )
                    workflow_info = questionnaire_json.get("workflow_info", {})
                    refined_query = workflow_info.get("refined_query")

                    response_data = {
                        "message": message,
                        "tools_mentioned": [],
                        "suggestions": [],
                        "examples": next_question.get("examples", []),
                        "tool_examples": next_question.get("tool_examples", []),
                        "progress_percentage": progress_percentage,
                    }

                    # Add refined query if it exists
                    if refined_query:
                        response_data["refined_query"] = refined_query

                    # Add refined query fields for Phase 4 (Refinement)
                    if current_phase == 4 and next_question.get("is_refined"):
                        response_data["is_refined"] = True
                        response_data["refined_query"] = next_question.get(
                            "refined_query", ""
                        )
                        logger.info(
                            f"🎯 Phase 4: Including refined query with is_refined flag"
                        )

                    return response_data
                else:
                    # Should not reach here
                    return {
                        "message": "Something went wrong. Let's start over.",
                        "tools_mentioned": [],
                        "suggestions": ["Start a new workflow"],
                    }

        except Exception as e:
            logger.error(f"❌ Error handling questionnaire answer: {e}", exc_info=True)
            return {
                "message": "Having trouble processing your answer. Try again.",
                "tools_mentioned": [],
                "suggestions": ["Try rephrasing your answer"],
            }

    async def _generate_workflow_from_nodes(
        self, conversation: ConversationSession
    ) -> Dict[str, Any]:
        """Generate workflow from conversation nodes."""
        try:
            # Create workflow structure from existing nodes
            workflow_id = str(uuid.uuid4())

            # Create edges between nodes (simple sequential connection)
            edges = []
            for i in range(len(conversation.workflow_nodes) - 1):
                edge_id = str(uuid.uuid4())  # Use UUID for edge ID
                edge = {
                    "id": edge_id,
                    "source": conversation.workflow_nodes[i]["id"],
                    "target": conversation.workflow_nodes[i + 1]["id"],
                    "type": "default",
                }
                edges.append(edge)

            # Update conversation with edges
            conversation.workflow_edges = edges
            await sync_to_async(conversation.save)()

            # Create complete workflow structure
            workflow = {
                "query": conversation.original_query,
                "nodes": conversation.workflow_nodes,
                "edges": edges,
            }

            # CRITICAL FIX: Apply UUID conversion to ensure no hardcoded IDs
            from .ai_agents.tools.ai_tool_recommender import AIToolRecommender

            recommender = AIToolRecommender()
            workflow = recommender._convert_hardcoded_ids_to_uuids(workflow)
            logger.info("✅ Applied UUID conversion to workflow from nodes")

            # Save workflow to database
            from workflow.models import Workflow

            workflow_obj = await sync_to_async(Workflow.objects.create)(
                id=workflow_id,
                metadata=workflow,
                owner=conversation.user,
                user_query=conversation.original_query,
                prompt=conversation.original_query,
            )

            # CRITICAL FIX: Save nodes and edges to Node/Edge tables!
            from .workflow_generation_service import WorkflowGenerationService

            workflow_service = WorkflowGenerationService()
            await workflow_service._save_to_node_edge_tables(workflow_obj, workflow)

            logger.info(
                f"✅ Saved {len(conversation.workflow_nodes)} nodes and {len(edges)} edges to Node/Edge tables for workflow {workflow_id}"
            )

            logger.info(
                f"✅ Created workflow {workflow_id} with {len(conversation.workflow_nodes)} nodes and {len(edges)} edges"
            )

            return {
                "workflow": workflow,
                "workflow_id": workflow_id,
                "nodes_count": len(conversation.workflow_nodes),
                "edges_count": len(edges),
            }

        except Exception as e:
            logger.error(f"Error generating workflow from nodes: {e}")
            raise e

    # ==================== QUERY REFINEMENT ENDPOINTS ====================

    @extend_schema(
        summary="Refine user query",
        description="Simple button-click query refinement - makes the query more specific and actionable",
        request=RefineQueryInitSerializer,
        responses={200: RefineQueryResponseSerializer},
        tags=["AI Tool Recommender - Query Refinement"],
    )
    @action(detail=False, methods=["post"], url_path="refine-query")
    async def refine_query(self, request):
        """Refine user query to make it more specific and actionable."""
        serializer = RefineQueryInitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        original_query = serializer.validated_data["original_prompt"]

        # Initialize query refinement service
        refinement_service = QueryRefinementService()

        # Refine the query
        result = await refinement_service.refine_query(original_query)

        logger.info(
            f"✅ Query refined: '{original_query}' → '{result['refined_query']}'"
        )

        # Save refinement to database for tracking and analytics
        refinement_session = await sync_to_async(RefineQuerySession.objects.create)(
            user=request.user if request.user.is_authenticated else None,
            original_prompt=original_query,
            refined_query=result["refined_query"],
            status="completed",
            refinement_history=[
                {
                    "timestamp": timezone.now().isoformat(),
                    "original": original_query,
                    "refined": result["refined_query"],
                    "explanation": result["explanation"],
                }
            ],
        )

        return Response(
            {
                "status": "success",
                "session_id": str(refinement_session.session_id),
                "original_query": result["original_query"],
                "refined_query": result["refined_query"],
                "explanation": result["explanation"],
            }
        )

    @extend_schema(
        summary="Get chat history",
        description="Retrieve the complete chat history for a workflow using workflow_id",
        request=ConversationChatSerializer,
        responses={200: ConversationalResponseSerializer},
        tags=["AI Tool Recommender - Conversation"],
    )
    @action(detail=False, methods=["post"], url_path="chat-history")
    async def get_chat_history(self, request):
        """Get the complete chat history for a workflow using workflow_id."""
        # Get workflow_id from request
        workflow_id = request.data.get("workflow_id")

        if not workflow_id:
            return Response(
                {"error": "Workflow ID is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Get or find conversation session based on workflow_id and user
        try:
            conversation = await sync_to_async(
                lambda: ConversationSession.objects.filter(
                    user=request.user,
                    current_context__workflow_id=str(workflow_id),
                    is_active=True,
                ).latest("updated_at")
            )()
        except ConversationSession.DoesNotExist:
            return Response(
                {"error": "No conversation found for this workflow"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get all chat messages for this session
        chat_messages = await sync_to_async(list)(
            ChatMessage.objects.filter(session=conversation).order_by("created_at")
        )

        return Response(
            {
                "workflow_id": str(workflow_id),
                "session_id": str(conversation.session_id),
                "original_query": conversation.original_query,
                "chat_history": conversation.chat_history,
                "workflow_preview": {
                    "nodes": conversation.workflow_nodes,
                    "edges": conversation.workflow_edges,
                },
                "chat_messages": [
                    {
                        "id": str(msg.id),
                        "user_message": msg.user_message,
                        "ai_response": msg.ai_response,
                        "message_type": msg.message_type,
                        "tools_mentioned": msg.tools_mentioned,
                        "workflow_changes": msg.workflow_changes,
                        "intent_analysis": msg.intent_analysis,
                        "created_at": msg.created_at.isoformat(),
                    }
                    for msg in chat_messages
                ],
                "total_messages": len(chat_messages),
                "total_nodes": len(conversation.workflow_nodes),
                "total_edges": len(conversation.workflow_edges),
                "workflow_ready": len(conversation.workflow_nodes) > 0,
            }
        )


@extend_schema_view(
    list=extend_schema(
        summary="List discovered tools",
        description="Get a list of discovered tools pending review",
        tags=["AI Tool Recommender - Discovery"],
    ),
)
class DiscoveredToolViewSet(viewsets.ModelViewSet):
    """ViewSet for discovered tools."""

    serializer_class = DiscoveredToolSerializer
    permission_classes = [IsAuthenticated]
    queryset = DiscoveredTool.objects.all()

    def get_queryset(self):
        """Get discovered tools with optional filters."""
        queryset = DiscoveredTool.objects.all()

        # Filter by status
        status_param = self.request.query_params.get("status")
        if status_param:
            queryset = queryset.filter(status=status_param)

        return queryset.order_by("-discovered_at")

    @extend_schema(
        summary="Approve a discovered tool",
        description="Approve a discovered tool and optionally add it to the main database",
        tags=["AI Tool Recommender - Discovery"],
    )
    @action(detail=True, methods=["post"], permission_classes=[IsAuthenticated])
    def approve(self, request, pk=None):
        """Approve a discovered tool."""
        discovered_tool = self.get_object()

        # Update status
        discovered_tool.status = "approved"
        discovered_tool.save()

        return Response(
            {
                "status": "success",
                "message": "Tool approved successfully",
                "tool_id": str(discovered_tool.id),
            }
        )

    @extend_schema(
        summary="Reject a discovered tool",
        description="Reject a discovered tool",
        tags=["AI Tool Recommender - Discovery"],
    )
    @action(detail=True, methods=["post"], permission_classes=[IsAuthenticated])
    def reject(self, request, pk=None):
        """Reject a discovered tool."""
        discovered_tool = self.get_object()

        # Update status
        discovered_tool.status = "rejected"
        discovered_tool.save()

        return Response(
            {
                "status": "success",
                "message": "Tool rejected",
                "tool_id": str(discovered_tool.id),
            }
        )


@extend_schema_view(
    list=extend_schema(
        summary="List workflow generations",
        description="Get a list of generated workflows",
        tags=["AI Tool Recommender - Workflows"],
    ),
)
class WorkflowGenerationViewSet(async_viewsets.ReadOnlyModelViewSet):
    """ViewSet for workflow generations with battle cards functionality."""

    serializer_class = WorkflowGenerationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get workflows for current user."""
        if self.request.user.is_superuser:
            return WorkflowGeneration.objects.all()
        return WorkflowGeneration.objects.filter(user=self.request.user)

    @extend_schema(
        summary="Compare tool alternatives (Battle Cards)",
        description="Find alternative tools for a specific tool in the workflow for battle cards comparison",
        request=ToolComparisonRequestSerializer,
        responses={200: ToolComparisonResultSerializer},
        tags=["AI Tool Recommender - Battle Cards"],
    )
    @action(detail=True, methods=["post"], url_path="compare-tool")
    async def compare_tool(self, request, pk=None):
        """Find alternative tools for battle cards comparison."""
        start_time = time.time()  # Add the missing start_time

        # Parse request data first
        serializer = ToolComparisonRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        node_id = serializer.validated_data["node_id"]
        max_results = serializer.validated_data.get("max_results", 8)
        include_explanations = serializer.validated_data.get(
            "include_explanations", True
        )

        # Determine the approach based on node_id format
        is_uuid_format = NodeLookupService.is_uuid_format(node_id)

        try:
            # Use the centralized workflow retrieval function
            (
                workflow_data,
                workflow_query,
                is_workflow_generation,
                original_workflow,
            ) = await get_workflow_with_nodes(pk)

            # Create mock workflow object for compatibility (only if needed)
            # Use original_workflow directly if it's a WorkflowGeneration
            if is_workflow_generation:
                workflow = original_workflow
            else:

                class MockWorkflow:
                    def __init__(self, workflow_data, query):
                        self.workflow_data = workflow_data
                        self.query = query
                        self.id = pk

                workflow = MockWorkflow(workflow_data, workflow_query)

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            # Use different search strategies based on node_id format
            nodes = workflow_data.get("nodes", [])
            target_node = None
            match_strategy = "none"

            # Debug: Log available nodes for troubleshooting
            logger.info(f"🔍 Available nodes in workflow {pk}: {len(nodes)} total")
            for i, node in enumerate(nodes):
                node_id_str = node.get("id", "N/A")
                node_type = node.get("type", "N/A")
                node_data = node.get("data", {})
                node_label = node_data.get("label", "N/A")
                original_id = node_data.get("original_id", "N/A")
                logger.info(
                    f"  Node {i}: ID={node_id_str[:8]}..., Type={node_type}, Label={node_label}, OriginalID={original_id}"
                )

            # Also log the id_mapping if available
            id_mapping = workflow_data.get("id_mapping", {})
            if id_mapping:
                logger.info(f"🔗 ID Mapping available: {len(id_mapping)} entries")
                # Log first few mappings for debugging
                sample_mappings = dict(list(id_mapping.items())[:5])
                logger.info(f"📋 Sample ID mappings: {sample_mappings}")
            else:
                logger.warning("⚠️ No ID mapping found in workflow_data")

            if is_uuid_format:
                # UUID format - use flexible lookup with all strategies
                target_node, match_strategy = WorkflowNodeMatcher.find_best_match(
                    nodes, node_id, workflow_data=workflow_data
                )

                # If not found, try database lookup (works for both WorkflowGeneration and Workflow)
                if not target_node:
                    try:
                        from workflow.models import Node

                        # Try to find node directly in database by UUID
                        db_node = await sync_to_async(
                            Node.objects.filter(id=node_id, workflow_id=pk).first
                        )()

                        if db_node:
                            # Convert database node to workflow node format
                            node_data = db_node.data or {}
                            target_node = {
                                "id": str(db_node.id),
                                "type": db_node.type or "tool",
                                "position": {
                                    "x": db_node.position_x or 0,
                                    "y": db_node.position_y or 0,
                                },
                                "positionAbsolute": {
                                    "x": db_node.position_absolute_x or 0,
                                    "y": db_node.position_absolute_y or 0,
                                },
                                "data": node_data,
                            }
                            match_strategy = "database_uuid_lookup"
                            logger.info(f"✅ Found node in database: {node_id}")

                            # Also add it to the nodes list for consistency
                            nodes.append(target_node)
                    except Exception as db_error:
                        logger.warning(
                            f"Database lookup failed for {node_id}: {db_error}"
                        )

            else:
                # Original format - try direct search first, then flexible lookup
                # Try direct ID match first (most common case for unsaved workflows)
                for node in nodes:
                    if node.get("id") == node_id:
                        target_node = node
                        match_strategy = "direct_id_match"
                        break

                # If not found, try flexible lookup as fallback
                if not target_node:
                    target_node, match_strategy = WorkflowNodeMatcher.find_best_match(
                        nodes, node_id, workflow_data=workflow_data
                    )

                # If still not found and we have UUID nodes, try to create dynamic mapping
                if (
                    not target_node
                    and len(
                        [
                            n
                            for n in nodes
                            if NodeLookupService.is_uuid_format(n.get("id", ""))
                        ]
                    )
                    > 0
                ):
                    # Parse the requested node_id to extract type and sequence
                    if "_" in node_id:
                        parts = node_id.split("_")
                        if len(parts) >= 2 and parts[-1].isdigit():
                            requested_type = "_".join(parts[:-1])
                            requested_sequence = int(parts[-1])

                            # Find nodes of the same type and create dynamic mapping
                            type_nodes = [
                                n for n in nodes if n.get("type") == requested_type
                            ]

                            # Sort by some consistent order (e.g., position or creation order)
                            # For now, use the order they appear in the list
                            if len(type_nodes) >= requested_sequence:
                                target_node = type_nodes[
                                    requested_sequence - 1
                                ]  # -1 because sequence is 1-based
                                match_strategy = "dynamic_mapping"

                    # Extract the sequence number from node_001 format
                    if "_" in node_id:
                        parts = node_id.split("_")
                        if len(parts) >= 2 and parts[-1].isdigit():
                            node_type = "_".join(parts[:-1])
                            sequence = int(parts[-1])

                            # Try to find a node by type and position
                            logger.info(
                                f"🔍 Looking for {node_type} node at position {sequence}"
                            )

                            # Filter nodes by type if possible
                            matching_type_nodes = [
                                n for n in nodes if n.get("type") == node_type
                            ]

                            if matching_type_nodes and sequence <= len(
                                matching_type_nodes
                            ):
                                # Use 1-based indexing (node_001 = first node of that type)
                                target_node = matching_type_nodes[sequence - 1]
                                match_strategy = "type_sequence_mapping"
                                logger.info(
                                    f"✅ Found node using type+sequence mapping: {node_type}[{sequence}] -> {target_node.get('id')}"
                                )
                            elif len(nodes) >= sequence:
                                # Fallback: use overall position in workflow
                                target_node = nodes[sequence - 1]
                                match_strategy = "position_fallback"
                                logger.info(match_strategy)
                                logger.info(
                                    f"✅ Found node using position fallback: position {sequence} -> {target_node.get('id')}"
                                )
                            else:
                                logger.warning(
                                    f"⚠️ Sequence {sequence} exceeds available nodes ({len(nodes)} total, {len(matching_type_nodes)} of type {node_type})"
                                )
                        else:
                            logger.warning(
                                f"⚠️ Could not parse sequence from node_id: {node_id}"
                            )
                    else:
                        logger.warning(
                            f"⚠️ Original format node_id doesn't contain underscore: {node_id}"
                        )

            if not target_node:
                # Get detailed information about available nodes for debugging
                available_nodes = NodeLookupService.get_available_nodes_info(nodes)

                # Log the error for debugging
                logger.error(f"❌ Node '{node_id}' not found in workflow {pk}")
                logger.error(f"📊 Workflow has {len(nodes)} nodes total")
                logger.error(
                    f"🔍 Available nodes: {[n.get('id', 'N/A')[:8] + '...' for n in nodes[:5]]}"
                )
                logger.error(f"🔍 Search strategy used: {match_strategy}")

                return Response(
                    {
                        "status": "error",
                        "message": f"Node with ID '{node_id}' not found in workflow",
                        "search_details": {
                            "requested_node_id": node_id,
                            "node_id_format": (
                                "uuid"
                                if is_uuid_format
                                else "original"
                                if NodeLookupService.is_original_format(node_id)
                                else "unknown"
                            ),
                            "search_approach": "uuid_hybrid"
                            if is_uuid_format
                            else "original_direct",
                            "match_strategy": match_strategy,
                            "total_nodes": len(nodes),
                            "id_mapping_available": bool(
                                workflow_data.get("id_mapping")
                            ),
                        },
                        "available_nodes": available_nodes,
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Initialize comparison service
            comparison_service = ToolComparisonService()

            # Find alternative tools
            comparison_result = await comparison_service.find_alternative_tools(
                original_tool_data=target_node.get("data", {}),
                node_id=node_id,  # Keep original requested ID for tracking
                max_results=max_results,
                include_explanations=include_explanations,
            )

            if comparison_result.get("status") == "error":
                return Response(
                    comparison_result, status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Save comparison to database
            search_time_ms = (time.time() - start_time) * 1000

            # Only save ToolComparison if it's a WorkflowGeneration (has the proper foreign key)
            if is_workflow_generation:
                await sync_to_async(ToolComparison.objects.create)(
                    user=request.user,
                    workflow=workflow,  # This is a real WorkflowGeneration instance
                    original_tool_node_id=node_id,
                    original_tool_data=target_node.get("data", {}),
                    comparison_query=comparison_result.get("comparison_query", ""),
                    alternative_tools=comparison_result.get("alternatives", []),
                    total_alternatives_found=comparison_result.get("total_found", 0),
                    search_time_ms=search_time_ms,
                )
                logger.info(
                    f"✅ Battle cards comparison saved to database for WorkflowGeneration {workflow.id}"
                )
            else:
                logger.info(
                    f"ℹ️ Skipping ToolComparison database save for regular Workflow {pk} (not a WorkflowGeneration)"
                )

            logger.info(
                f"Battle cards comparison completed: {comparison_result.get('total_found', 0)} alternatives found for {target_node.get('data', {}).get('label', 'Unknown Tool')}"
            )

            return Response(comparison_result)

        except Exception as e:
            logger.error(f"Error in compare_tool: {e}")
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                    "node_id": node_id,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Replace tool in workflow",
        description="Replace a specific tool in the workflow with an alternative tool",
        request=WorkflowToolReplaceSerializer,
        responses={200: WorkflowUpdateResultSerializer},
        tags=["AI Tool Recommender - Battle Cards"],
    )
    @action(detail=True, methods=["patch"], url_path="replace-tool")
    async def replace_tool(self, request, pk=None):
        """Replace a tool in the workflow with an alternative."""
        # Parse request data first
        serializer = WorkflowToolReplaceSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        node_id = serializer.validated_data["node_id"]
        new_tool_data = serializer.validated_data["new_tool_data"]
        preserve_connections = serializer.validated_data.get(
            "preserve_connections", True
        )

        # Determine the approach based on node_id format
        is_uuid_format = NodeLookupService.is_uuid_format(node_id)

        try:
            if is_uuid_format:
                # UUID format - use the hybrid approach with database retrieval
                (
                    workflow_data,
                    workflow_query,
                    is_workflow_generation,
                    original_workflow,
                ) = await get_workflow_with_nodes(pk)

            else:
                # Original format - try direct WorkflowGeneration first, then fall back to hybrid approach
                try:
                    from .models import WorkflowGeneration

                    workflow_generation = await sync_to_async(
                        WorkflowGeneration.objects.get
                    )(id=pk)
                    workflow_data = workflow_generation.workflow_data
                    workflow_query = workflow_generation.query
                    is_workflow_generation = True
                    original_workflow = workflow_generation

                except Exception as wg_error:
                    logger.error(f"Error in replace_tool: {wg_error}")
                    # Fall back to hybrid approach if WorkflowGeneration doesn't exist
                    (
                        workflow_data,
                        workflow_query,
                        is_workflow_generation,
                        original_workflow,
                    ) = await get_workflow_with_nodes(pk)

            # Create mock workflow object for compatibility
            class MockWorkflow:
                def __init__(self, workflow_data, workflow_obj):
                    self.workflow_data = workflow_data
                    self.id = pk
                    self._original_workflow = workflow_obj

            workflow = MockWorkflow(workflow_data, original_workflow)

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            # If UUID format and node not found in workflow_data, try database lookup
            nodes = workflow_data.get("nodes", [])
            node_found_in_data = any(node.get("id") == node_id for node in nodes)

            if is_uuid_format and not node_found_in_data:
                logger.info(
                    f"🔍 UUID node {node_id} not found in workflow_data, trying database lookup..."
                )

                # Strategy 1: Check id_mapping in workflow_data metadata
                id_mapping = workflow_data.get("id_mapping", {})
                if node_id in id_mapping:
                    mapped_id = id_mapping[node_id]
                    logger.info(f"✅ Found UUID in id_mapping: {node_id} -> {mapped_id}")
                    node_id = mapped_id
                    node_found_in_data = any(
                        node.get("id") == node_id for node in nodes
                    )

                # Strategy 2: Database lookup (works for both WorkflowGeneration and Workflow)
                if not node_found_in_data:
                    try:
                        from workflow.models import Node

                        # Try to find node directly in database
                        db_node = await sync_to_async(
                            Node.objects.filter(id=node_id, workflow_id=pk).first
                        )()

                        if db_node:
                            # Convert database node to workflow node format
                            node_data = db_node.data or {}
                            db_workflow_node = {
                                "id": str(db_node.id),
                                "type": db_node.type or "tool",
                                "position": {
                                    "x": db_node.position_x or 0,
                                    "y": db_node.position_y or 0,
                                },
                                "positionAbsolute": {
                                    "x": db_node.position_absolute_x or 0,
                                    "y": db_node.position_absolute_y or 0,
                                },
                                "data": node_data,
                            }
                            # Add to workflow_data nodes so the service can find it
                            workflow_data["nodes"].append(db_workflow_node)
                            logger.info(
                                f"✅ Found node in database and added to workflow_data: {node_id}"
                            )
                            node_found_in_data = True
                    except Exception as db_error:
                        logger.warning(
                            f"Database lookup failed for {node_id}: {db_error}"
                        )

                # Strategy 3: Use WorkflowNodeMatcher as final fallback
                if not node_found_in_data:
                    target_node, match_strategy = WorkflowNodeMatcher.find_best_match(
                        nodes, node_id, workflow_data=workflow_data
                    )
                    if target_node:
                        logger.info(
                            f"✅ Found node using WorkflowNodeMatcher: {match_strategy}"
                        )
                        node_id = target_node.get("id")
                        node_found_in_data = True

                # If still not found, return helpful error
                if not node_found_in_data:
                    available_nodes = NodeLookupService.get_available_nodes_info(nodes)
                    return Response(
                        {
                            "status": "error",
                            "message": f"Node with ID '{node_id}' not found in workflow. Available node IDs: {[n['id'] for n in available_nodes]}",
                            "requested_node_id": node_id,
                            "available_nodes": [
                                {
                                    "id": node_info["id"],
                                    "label": node_info["label"],
                                    "type": node_info["type"],
                                }
                                for node_info in available_nodes
                            ],
                            "workflow": workflow_data,
                        },
                        status=status.HTTP_404_NOT_FOUND,
                    )

            # Initialize workflow update service
            update_service = WorkflowUpdateService()

            # Replace the tool in the workflow
            update_result = await update_service.replace_tool_in_workflow(
                workflow_data=workflow_data,
                node_id=node_id,
                new_tool_data=new_tool_data,
                preserve_connections=preserve_connections,
            )

            if update_result.get("status") == "error":
                return Response(update_result, status=status.HTTP_400_BAD_REQUEST)

            # Validate the updated workflow
            validation_result = await update_service.validate_workflow_structure(
                update_result["workflow"]
            )

            # Update the workflow in the appropriate model
            await update_workflow_with_nodes(
                update_result["workflow"], is_workflow_generation, original_workflow
            )

            logger.info(
                f"Tool replaced in workflow {pk}: {node_id} -> {new_tool_data.get('Title') or new_tool_data.get('title', 'Unknown Tool')}"
            )

            # Include validation results in response
            response_data = update_result.copy()
            response_data["validation"] = validation_result

            return Response(response_data)

        except Exception as e:
            logger.error(f"Error in replace_tool: {e}")
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                    "workflow": workflow.workflow_data,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema_view(
    list=extend_schema(
        summary="List tool comparisons",
        description="Get a list of tool comparisons for battle cards",
        tags=["AI Tool Recommender - Battle Cards"],
    ),
)
class ToolComparisonViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for tool comparisons (Battle Cards history)."""

    serializer_class = ToolComparisonSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get tool comparisons for current user."""
        if self.request.user.is_superuser:
            return ToolComparison.objects.all()
        return ToolComparison.objects.filter(user=self.request.user)

    @extend_schema(
        summary="Get comparison details",
        description="Get detailed information about a specific tool comparison",
        tags=["AI Tool Recommender - Battle Cards"],
    )
    @action(detail=True, methods=["get"])
    def details(self, request, pk=None):
        """Get detailed comparison information."""
        comparison = self.get_object()
        serializer = self.get_serializer(comparison)

        # Add additional metadata
        data = serializer.data
        data["alternative_count"] = len(comparison.alternative_tools)
        data["has_explanations"] = any(
            "comparison" in tool for tool in comparison.alternative_tools
        )

        return Response(data)


@extend_schema_view(
    list=extend_schema(
        summary="List background tasks",
        description="Get a list of background tasks",
        tags=["AI Tool Recommender - Background Tasks"],
    ),
)
class BackgroundTaskViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for background tasks."""

    serializer_class = BackgroundTaskSerializer
    permission_classes = [IsAuthenticated]
    queryset = BackgroundTask.objects.all()

    @extend_schema(
        summary="Get task status",
        description="Get the status of a specific background task",
        tags=["AI Tool Recommender - Background Tasks"],
    )
    @action(detail=True, methods=["get"])
    def task_status(self, request, pk=None):
        """Get status of a specific task."""
        task = self.get_object()
        serializer = self.get_serializer(task)
        return Response(serializer.data)


@extend_schema(
    summary="Explain a tool",
    description="Generate an explanation for an AI tool based on user query",
    request=ExplainToolSerializer,
    tags=["AI Tool Recommender - Utilities"],
)
class ExplainToolView(async_viewsets.ViewSet):
    """ViewSet for tool explanations with async support."""

    permission_classes = [AllowAny]

    @action(detail=False, methods=["post"])
    async def explain(self, request):
        """Explain a tool."""
        serializer = ExplainToolSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            explanation = await generate_tool_explanation(
                json_object=serializer.validated_data["json_object"],
                query=serializer.validated_data["query"],
            )

            return Response(
                {
                    "status": "success",
                    "explanation": explanation,
                }
            )

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema(tags=["AI Tool Recommender - Refined Queries"])
class RefinedQueryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for retrieving refined queries by workflow_id.

    Refined queries are generated after completing the 3-phase questionnaire
    and contain detailed Markdown-formatted workflow requirements.
    """

    queryset = RefinedQuery.objects.all()
    serializer_class = RefinedQuerySerializer
    permission_classes = [AllowAny]
    lookup_field = "workflow_id"

    @extend_schema(
        summary="List all refined queries",
        description="Get a paginated list of all refined queries. "
        "Optionally filter by workflow_id using query parameter.",
        parameters=[
            {
                "name": "workflow_id",
                "in": "query",
                "description": "Filter by workflow_id (UUID)",
                "required": False,
                "schema": {"type": "string", "format": "uuid"},
            }
        ],
        responses={200: RefinedQuerySerializer(many=True)},
        tags=["AI Tool Recommender - Refined Queries"],
    )
    def list(self, request, *args, **kwargs):
        """List all refined queries with optional filtering."""
        return super().list(request, *args, **kwargs)

    def get_queryset(self):
        """Filter by workflow_id if provided."""
        queryset = RefinedQuery.objects.all()
        workflow_id = self.request.query_params.get("workflow_id")
        if workflow_id:
            queryset = queryset.filter(workflow_id=workflow_id)
        return queryset.order_by("-created_at")

    @extend_schema(
        summary="Get refined query by workflow_id",
        description="Retrieve the refined query (in Markdown format) for a specific workflow. "
        "The refined query is generated after completing Phase 1 and Phase 2 of the questionnaire.",
        responses={
            200: RefinedQuerySerializer,
            404: {"description": "Refined query not found for this workflow_id"},
        },
        tags=["AI Tool Recommender - Refined Queries"],
    )
    @action(
        detail=False, methods=["get"], url_path="by-workflow/(?P<workflow_id>[^/.]+)"
    )
    def by_workflow(self, request, workflow_id=None):
        """Get refined query by workflow_id."""
        try:
            refined_query = RefinedQuery.objects.filter(workflow_id=workflow_id).first()
            if not refined_query:
                return Response(
                    {
                        "status": "error",
                        "message": "Refined query not found for this workflow_id",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            serializer = self.get_serializer(refined_query)
            return Response(
                {
                    "status": "success",
                    "data": serializer.data,
                }
            )

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# ==================== WORKFLOW IMPLEMENTATION GUIDE ENDPOINTS ====================


@extend_schema_view(
    create=extend_schema(
        summary="Generate workflow implementation guide",
        description="Generate or update a comprehensive implementation guide for a workflow. "
        "This endpoint takes a workflow_id and creates a detailed, step-by-step guide "
        "on how to implement the workflow with all its tools and connections.",
        request=WorkflowImplementationRequestSerializer,
        responses={
            200: WorkflowImplementationResponseSerializer,
            400: {"description": "Bad Request"},
            404: {"description": "Workflow not found"},
            500: {"description": "Internal Server Error"},
        },
        tags=["AI Tool Recommender - Implementation Guide"],
    ),
    retrieve=extend_schema(
        summary="Get workflow implementation guide",
        description="Retrieve an existing implementation guide for a workflow.",
        responses={
            200: WorkflowImplementationGuideSerializer,
            404: {"description": "Implementation guide not found"},
        },
        tags=["AI Tool Recommender - Implementation Guide"],
    ),
    list=extend_schema(
        summary="List workflow implementation guides",
        description="Get a list of all implementation guides for the current user.",
        responses={200: WorkflowImplementationGuideSerializer(many=True)},
        tags=["AI Tool Recommender - Implementation Guide"],
    ),
)
class WorkflowImplementationGuideViewSet(async_viewsets.ModelViewSet):
    """ViewSet for workflow implementation guides."""

    serializer_class = WorkflowImplementationGuideSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = "workflow_id"

    def get_queryset(self):
        """Filter implementation guides by current user."""
        return WorkflowImplementationGuide.objects.filter(user=self.request.user)

    async def retrieve(self, request, *args, **kwargs):
        """Retrieve implementation guide by workflow_id."""
        try:
            workflow_id = kwargs.get("workflow_id") or kwargs.get("pk")
            if not workflow_id:
                return Response(
                    {"status": "error", "message": "workflow_id is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            user = request.user
            implementation_service = WorkflowImplementationService()

            # Get existing implementation guide
            guide_data = await implementation_service.get_implementation_guide(
                workflow_id=workflow_id, user=user
            )

            if not guide_data:
                return Response(
                    {
                        "status": "error",
                        "message": "Implementation guide not found for this workflow",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Get the guide object for serializer
            guide_obj = await sync_to_async(
                lambda: WorkflowImplementationGuide.objects.filter(
                    workflow_id=workflow_id, user=user
                ).first()
            )()

            if not guide_obj:
                return Response(
                    {
                        "status": "error",
                        "message": "Implementation guide not found for this workflow",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            serializer = self.get_serializer(guide_obj)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"❌ Error retrieving implementation guide: {e}", exc_info=True)
            return Response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Generate implementation guide",
        description="Generate or update implementation guide for a workflow",
        request=WorkflowImplementationRequestSerializer,
        responses={
            200: WorkflowImplementationResponseSerializer,
            400: {"description": "Bad Request"},
            404: {"description": "Workflow not found"},
        },
        tags=["AI Tool Recommender - Implementation Guide"],
    )
    async def create(self, request, *args, **kwargs):
        """Generate or update implementation guide for a workflow."""
        try:
            # Validate request data
            request_serializer = WorkflowImplementationRequestSerializer(
                data=request.data
            )
            if not request_serializer.is_valid():
                return Response(
                    {
                        "status": "error",
                        "errors": request_serializer.errors,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            workflow_id = request_serializer.validated_data["workflow_id"]
            user = request.user

            logger.info(f"🚀 Generating implementation guide for workflow {workflow_id}")

            # Get workflow data using the existing helper function
            try:
                (
                    workflow_data,
                    workflow_query,
                    is_workflow_generation,
                    original_workflow,
                ) = await get_workflow_with_nodes(str(workflow_id))
            except Exception as e:
                logger.error(f"❌ Workflow not found: {e}")
                return Response(
                    {
                        "status": "error",
                        "error": f"Workflow with ID '{workflow_id}' not found",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Initialize implementation service
            implementation_service = WorkflowImplementationService()

            # Generate or update implementation guide
            # Wrap in try-except to handle executor shutdown errors gracefully
            try:
                result = await implementation_service.generate_or_update_implementation_guide(
                    workflow_id=str(workflow_id),
                    user=user,
                    workflow_data=workflow_data,
                    workflow_query=workflow_query,
                )
            except RuntimeError as e:
                if "shutdown" in str(e).lower() or "cannot schedule" in str(e).lower():
                    logger.error(
                        f"❌ Executor shutdown error during guide generation: {e}",
                        exc_info=True,
                    )
                    return Response(
                        {
                            "status": "error",
                            "error": "Service temporarily unavailable due to system restart. Please try again in a moment.",
                            "implementation_guide": None,
                        },
                        status=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )
                raise

            # Return appropriate status code based on result
            response_status = status.HTTP_200_OK
            if result["status"] == "created":
                response_status = status.HTTP_201_CREATED
            elif result["status"] == "error":
                response_status = status.HTTP_500_INTERNAL_SERVER_ERROR

            return Response(result, status=response_status)

        except Exception as e:
            logger.error(f"❌ Error in implementation guide generation: {e}")
            return Response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Get implementation guide by workflow ID",
        description="Retrieve existing implementation guide for a specific workflow",
        responses={
            200: WorkflowImplementationGuideSerializer,
            404: {"description": "Implementation guide not found"},
        },
        tags=["AI Tool Recommender - Implementation Guide"],
    )
    @action(
        detail=False, methods=["get"], url_path="by-workflow/(?P<workflow_id>[^/.]+)"
    )
    async def by_workflow(self, request, workflow_id=None):
        """Get implementation guide by workflow_id."""
        try:
            user = request.user
            implementation_service = WorkflowImplementationService()

            # Get existing implementation guide
            guide_data = await implementation_service.get_implementation_guide(
                workflow_id=workflow_id, user=user
            )

            if not guide_data:
                return Response(
                    {
                        "status": "error",
                        "message": "Implementation guide not found for this workflow",
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            return Response(
                {
                    "status": "success",
                    "data": guide_data,
                }
            )

        except Exception as e:
            logger.error(f"❌ Error retrieving implementation guide: {e}")
            return Response(
                {
                    "status": "error",
                    "error": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

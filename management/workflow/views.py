"""Views for managing workflows."""

import logging
import uuid  # Import the uuid module

import openai
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema
from envs.env_loader import env_loader
from langchain_openai import ChatOpenAI
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ai_tool_recommender.models import ChatMessage, ConversationSession
from configurations.pagination import CustomPagination

from .models import Edge, Node, Workflow  # Import the Workflow, Node, and Edge models
from .serializers import (
    WorkflowSaveSerializer,
    WorkflowSaveWithIdSerializer,
    WorkflowSerializer,
)
from .transform_data import transform_workflow_data
from .utils.node_lookup import NodeLookupService

logger = logging.getLogger(__name__)

# Initialize OpenAI API
openai.api_key = env_loader.openai_api_key

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=env_loader.openai_api_key)


class WorkflowViewSet(viewsets.ModelViewSet):
    """A viewset for viewing and editing workflows."""

    queryset = Workflow.objects.all()
    serializer_class = WorkflowSerializer
    pagination_class = CustomPagination

    def get_queryset(self):
        """Return workflows belonging to the current user only."""
        return Workflow.objects.filter(owner=self.request.user)

    def get_serializer_class(self):
        """Return the appropriate serializer class based on the action."""
        if self.action == "save_workflow":
            return WorkflowSaveWithIdSerializer
        return super().get_serializer_class()

    def perform_create(self, serializer):
        """Save the workflow with the current user as the owner."""
        serializer.save(owner=self.request.user)

    def create(self, request, *args, **kwargs):
        """Create a new workflow."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(
            serializer.data, status=status.HTTP_201_CREATED, headers=headers
        )

    def destroy(self, request, *args, **kwargs):
        """Delete a workflow."""
        try:
            workflow = self.get_object()
            workflow.delete()
            return Response(
                {"message": "Workflow deleted successfully"},
                status=status.HTTP_200_OK,
            )
        except Workflow.DoesNotExist:
            return Response(
                {"message": "Workflow not found"}, status=status.HTTP_404_NOT_FOUND
            )

    def retrieve(self, request, *args, **kwargs):
        """Retrieve a specific workflow."""
        try:
            workflow = self.get_object()
            serializer = self.get_serializer(workflow)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Workflow.DoesNotExist:
            return Response(
                {"error": "Workflow not found"}, status=status.HTTP_404_NOT_FOUND
            )

    def list(self, request, *args, **kwargs):
        """List all workflows for the current user."""
        queryset = self.get_queryset()
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=["post"])
    def save_workflow(self, request, *args, **kwargs):
        """Save or update a workflow with nodes and edges."""
        print("=== SAVE WORKFLOW DEBUG ===")
        print("request.data:", request.data)

        workflow_data = request.data.get("workflow", {})
        print("workflow_data:", workflow_data)

        tools_data = workflow_data.get("tools", {})
        print("tools_data:", tools_data)

        workflow_id = request.data.get("workflow_id", "")
        print("workflow_id:", workflow_id)

        # Remove existing nodes and edges for the specific workflow
        Node.objects.filter(workflow_id=workflow_id).delete()
        Edge.objects.filter(workflow_id=workflow_id).delete()

        # Prepare workflow data - handle multiple possible data structures
        tools_data = workflow_data.get("tools", {})

        # Try different possible structures
        if tools_data and ("nodes" in tools_data or "edges" in tools_data):
            # Structure: workflow.tools.nodes/edges
            nodes = tools_data.get("nodes", [])
            edges = tools_data.get("edges", [])
            metadata = tools_data.get("metadata", {})
        elif "nodes" in workflow_data or "edges" in workflow_data:
            # Structure: workflow.nodes/edges (direct)
            nodes = workflow_data.get("nodes", [])
            edges = workflow_data.get("edges", [])
            metadata = workflow_data.get("metadata", {})
        else:
            # Fallback: empty
            nodes = []
            edges = []
            metadata = {}

        workflow_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": metadata,
        }

        print("final extracted nodes:", nodes)
        print("final extracted edges:", edges)
        print("final metadata:", metadata)

        # Ensure 'metadata' is present in the workflow data
        metadata = workflow_data.get("metadata", {})
        nodes = workflow_data.get("nodes", [])
        edges = workflow_data.get("edges", [])

        # Check if the workflow already exists
        workflow, created = Workflow.objects.update_or_create(
            id=workflow_id,
            defaults={
                "metadata": metadata,
                "owner": self.request.user,
            },
        )

        # Add new nodes with new UUIDs and preserve original IDs
        node_id_map = {}
        enhanced_nodes = []
        print(f"nodes: {enhanced_nodes}")
        # Create mapping and enhance nodes with original ID preservation
        for node_data in nodes:
            original_id = node_data["id"]

            # Check if the ID is already a UUID - if so, keep it
            try:
                uuid.UUID(original_id)
                # It's already a valid UUID, use it as-is
                new_uuid = original_id
                logger.info(f"✅ Keeping existing UUID: {original_id}")
            except ValueError:
                # It's not a UUID, generate a new one
                new_uuid = str(uuid.uuid4())
                logger.info(f"🔄 Generated new UUID for {original_id}: {new_uuid}")

            node_id_map[original_id] = new_uuid

            # Create enhanced node data with original ID preserved
            enhanced_node_data = (
                node_data["data"].copy() if node_data.get("data") else {}
            )

            # Store original ID mapping for future lookups
            enhanced_node_data["original_id"] = original_id
            enhanced_node_data["id_format"] = (
                "original"
                if NodeLookupService.is_original_format(original_id)
                else "custom"
            )

            # If original ID follows pattern (e.g., node_001), store sequence info
            if NodeLookupService.is_original_format(original_id) and "_" in original_id:
                parts = original_id.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    enhanced_node_data["node_type"] = "_".join(parts[:-1])
                    enhanced_node_data["sequence"] = int(parts[-1])

            # Create node with enhanced data
            Node.objects.create(
                id=new_uuid,
                type=node_data["type"],
                position_x=node_data.get("position", {}).get("x", 0),
                position_y=node_data.get("position", {}).get("y", 0),
                data=enhanced_node_data,
                workflow=workflow,
                dragging=node_data.get("dragging", False),
                height=node_data.get("height"),
                width=node_data.get("width"),
                position_absolute_x=node_data.get("positionAbsolute", {}).get("x", 0),
                position_absolute_y=node_data.get("positionAbsolute", {}).get("y", 0),
                selected=node_data.get("selected", False),
            )

            print(f"✅ Saved node with ID mapping: {original_id} -> {new_uuid}")

        # Add new edges with updated source/target references
        for edge_data in edges:
            source_id = edge_data["source"]
            target_id = edge_data["target"]

            # Map source and target IDs using the node_id_map
            mapped_source = node_id_map.get(
                source_id, source_id
            )  # Use original if not in map
            mapped_target = node_id_map.get(
                target_id, target_id
            )  # Use original if not in map

            logger.info(
                f"🔗 Creating edge: {source_id} -> {target_id} (mapped: {mapped_source} -> {mapped_target})"
            )

            Edge.objects.create(
                id=str(uuid.uuid4()),
                source=mapped_source,
                target=mapped_target,
                type=edge_data.get("type", "default"),
                workflow=workflow,
            )

        # Return the saved or updated workflow data
        return Response(
            {
                "status": "success",
                "message": (
                    "Workflow saved successfully"
                    if created
                    else "Workflow updated successfully"
                ),
                "workflow": workflow.id,
            },
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["get"])
    def get_workflow_data(self, request, *args, **kwargs):
        """Retrieve the data for a specific workflow including chat history."""
        workflow = self.get_object()
        serializer = WorkflowSaveSerializer(workflow)
        transformed_data = transform_workflow_data(serializer.data)

        # Ensure all nodes have recommendation_reason (generate if missing)
        from ai_tool_recommender.workflow_generation_service import (
            WorkflowGenerationService,
        )

        workflow_service = WorkflowGenerationService()

        for node in transformed_data.get("nodes", []):
            node_data = node.get("data", {})

            # If recommendation_reason is missing, generate it
            if "recommendation_reason" not in node_data or not node_data.get(
                "recommendation_reason"
            ):
                tool_name = node_data.get("label", "Tool")
                description = node_data.get("description", "")
                features = node_data.get("features", [])
                sequence = node_data.get("sequence", 1)

                recommendation_reason = (
                    workflow_service._generate_node_recommendation_reason(
                        tool_name, description, features, sequence
                    )
                )

                node_data["recommendation_reason"] = recommendation_reason

        # Fetch chat messages and progress from ConversationSession if exists
        progress_data = None
        conversation = None

        try:
            # Look for conversation session associated with this workflow
            conversation = (
                ConversationSession.objects.filter(
                    user=request.user,
                    current_context__workflow_id=str(workflow.id),
                    is_active=True,
                )
                .order_by("-updated_at")
                .first()
            )

            # Calculate progress from questionnaire if available
            if conversation and conversation.questionnaire_json:
                progress_data = self._calculate_questionnaire_progress(
                    conversation.questionnaire_json
                )
        except Exception as e:
            # If no conversation found or error, just continue
            print(f"Error fetching conversation: {e}")

        # Add progress percentage to the response
        if progress_data:
            transformed_data["progress"] = progress_data
        else:
            # Calculate basic workflow completion progress
            transformed_data["progress"] = self._calculate_workflow_progress(
                workflow, conversation
            )

        # Add refined query from database if exists
        try:
            from ai_tool_recommender.models import RefinedQuery

            refined_query_obj = RefinedQuery.objects.filter(
                workflow_id=workflow.id
            ).first()

            if refined_query_obj:
                transformed_data[
                    "refined_query_from_db"
                ] = refined_query_obj.refined_query
                transformed_data["is_refined"] = True
                # Set progress to 100 when refined query exists (refinement complete)
                transformed_data["progress"] = 100
            else:
                transformed_data["refined_query_from_db"] = None
                transformed_data["is_refined"] = False
        except Exception as e:
            print(f"Error fetching refined query: {e}")
            transformed_data["refined_query_from_db"] = None
            transformed_data["is_refined"] = False

        # Add implementation guide from database if exists
        try:
            from ai_tool_recommender.models import WorkflowImplementationGuide

            implementation_guide_obj = WorkflowImplementationGuide.objects.filter(
                workflow_id=workflow.id, user=request.user
            ).first()

            if implementation_guide_obj:
                transformed_data["implementation_guide"] = {
                    "id": str(implementation_guide_obj.id),
                    "content": implementation_guide_obj.implementation_guide,
                    "status": implementation_guide_obj.status,
                    "tools_count": implementation_guide_obj.tools_count,
                    "generation_time_ms": implementation_guide_obj.generation_time_ms,
                    "created_at": implementation_guide_obj.created_at.isoformat(),
                    "updated_at": implementation_guide_obj.updated_at.isoformat(),
                    "error_message": implementation_guide_obj.error_message,
                }
                transformed_data["has_implementation_guide"] = True
            else:
                transformed_data["implementation_guide"] = None
                transformed_data["has_implementation_guide"] = False
        except Exception as e:
            print(f"Error fetching implementation guide: {e}")
            transformed_data["implementation_guide"] = None
            transformed_data["has_implementation_guide"] = False

        # Add cached consultants from database if exists
        try:
            from consultant_recommender.models import WorkflowConsultants

            workflow_consultants = WorkflowConsultants.objects.filter(
                workflow_id=str(workflow.id), user=request.user
            ).first()

            if workflow_consultants:
                transformed_data["consultants"] = {
                    "data": workflow_consultants.consultants_data,
                    "count": workflow_consultants.consultants_count,
                    "query": workflow_consultants.query,
                    "user_work_description": workflow_consultants.user_work_description,
                    "search_metadata": workflow_consultants.search_metadata,
                    "cached_at": workflow_consultants.updated_at.isoformat(),
                    "created_at": workflow_consultants.created_at.isoformat(),
                }
                transformed_data["has_consultants"] = True
            else:
                transformed_data["consultants"] = None
                transformed_data["has_consultants"] = False
        except Exception as e:
            print(f"Error fetching workflow consultants: {e}")
            transformed_data["consultants"] = None
            transformed_data["has_consultants"] = False

        # Fetch all chat messages with agent tags and format as structured chat_history
        try:
            if conversation:
                chat_messages = ChatMessage.objects.filter(
                    session=conversation
                ).order_by("created_at")

                # Format chat_history as structured messages with user and agent messages
                chat_history = []
                for msg in chat_messages:
                    # Add user message
                    user_agent = (
                        msg.user_responded_to_agent.strip()
                        if msg.user_responded_to_agent
                        else None
                    )
                    # Convert "unknown" to None
                    if user_agent and user_agent.lower() == "unknown":
                        user_agent = None
                    chat_history.append(
                        {
                            "type": "user",
                            "message": msg.user_message,
                            "agent": user_agent,  # Which agent user was responding to (None if empty/unknown)
                            "timestamp": msg.created_at.isoformat(),
                        }
                    )
                    # Add agent response
                    chat_history.append(
                        {
                            "type": "agent",
                            "message": msg.ai_response,
                            "agent": msg.agent_name,  # Which agent responded
                            "message_type": msg.message_type,
                            "tools_mentioned": msg.tools_mentioned,
                            "workflow_changes": msg.workflow_changes,
                            "timestamp": msg.created_at.isoformat(),
                        }
                    )

                transformed_data["chat_history"] = chat_history
                transformed_data["total_chat_messages"] = len(chat_messages)
            else:
                transformed_data["chat_history"] = []
                transformed_data["total_chat_messages"] = 0
        except Exception as e:
            print(f"Error fetching chat messages: {e}")
            transformed_data["chat_history"] = []
            transformed_data["total_chat_messages"] = 0

        return Response(transformed_data, status=status.HTTP_200_OK)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="order",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description="Sort order for workflows",
                enum=["latest", "oldest"],
                default="latest",
            ),
        ],
        responses={
            200: OpenApiResponse(
                description="Successfully retrieved sorted workflows",
                response=WorkflowSerializer(many=True),
            ),
        },
        description=(
            "Get workflows sorted by updated timestamp (latest or oldest first)"
        ),
    )
    @action(detail=False, methods=["get"])
    def sorted_workflows(self, request):
        """Get workflows sorted by updated_at timestamp."""
        sort_order = request.query_params.get("order", "latest")

        # Start with the base queryset that's already filtered by owner
        queryset = self.get_queryset()

        # Apply sorting
        if sort_order.lower() == "oldest":
            queryset = queryset.order_by("updated_at")
        else:
            queryset = queryset.order_by("-updated_at")

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="q",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                description="Search query string",
                required=True,
            ),
            OpenApiParameter(
                name="threshold",
                type=OpenApiTypes.FLOAT,
                location=OpenApiParameter.QUERY,
                description="Minimum similarity threshold (0.0 to 1.0)",
                required=False,
                default=0.3,
            ),
        ],
        responses={200: WorkflowSerializer(many=True)},
        description="Search workflows using fuzzy matching with relevance scoring",
    )
    @action(detail=False, methods=["get"])
    def search(self, request):
        """Search workflows using fuzzy matching with relevance scoring."""
        query = request.query_params.get("q", "").strip()
        threshold = float(request.query_params.get("threshold", 0.3))

        if not query:
            return Response(
                {"error": "Search query is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not (0 <= threshold <= 1):
            return Response(
                {"error": "Threshold must be between 0 and 1"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        queryset = Workflow.search(query)
        queryset = queryset.filter(owner=request.user)

        page = self.paginate_queryset(queryset)

        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response_data = serializer.data

            # Add relevance scores to the response with safe handling of None values
            for item, obj in zip(response_data, page):
                item["relevance"] = {
                    "total_similarity": round(float(obj.total_similarity or 0.0), 3),
                    "rank": round(float(obj.rank or 0.0), 3),
                }

            return self.get_paginated_response(response_data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def _calculate_questionnaire_progress(self, questionnaire_json):
        """
        Calculate progress percentage based on questions answered.

        Phase 1: 5 questions (0-50%)
        Phase 2: 5 questions (50-100%)
        Phase 3: 1 question (stays at 100%)

        Each question = 10% (10 questions total in Phase 1 + Phase 2)
        """
        try:
            # Count total questions answered across all phases
            total_answered = 0

            # Phase 1: 5 questions (each = 10%)
            phase_1 = questionnaire_json.get("phase_1", {})
            phase_1_answered = len(phase_1.get("answers", {}))
            total_answered += phase_1_answered

            # Phase 2: 5 questions (each = 10%)
            phase_2 = questionnaire_json.get("phase_2", {})
            phase_2_answered = len(phase_2.get("answers", {}))
            total_answered += phase_2_answered

            # Phase 3: 1 question (doesn't add to percentage, stays at 100%)
            # We don't count Phase 3 as it's just confirmation

            # Each question = 10% (10 questions total in Phase 1 + Phase 2)
            percentage = min(total_answered * 10, 100)

            return percentage
        except Exception as e:
            print(f"Error calculating questionnaire progress: {e}")
            return 0

    def _calculate_workflow_progress(self, workflow, conversation):
        """Calculate basic workflow completion progress based on chat messages."""
        try:
            # If no conversation or chat history, return 0
            if not conversation or not conversation.chat_history:
                return 0

            # Count the number of user messages in chat_history
            # Each item in chat_history is a dict with "user", "ai", "timestamp"
            user_message_count = len(conversation.chat_history)

            # Calculate progress: (number of messages - 1) * 10
            # 2 messages = 10%, 3 messages = 20%, etc.
            progress_percentage = (user_message_count - 1) * 10

            # Cap at 100%
            progress_percentage = min(progress_percentage, 100)

            # Ensure it's not negative
            progress_percentage = max(progress_percentage, 0)

            return round(progress_percentage, 1)
        except Exception as e:
            print(f"Error calculating workflow progress: {e}")
            return 0

    @extend_schema(
        responses={
            200: OpenApiResponse(
                description="Successfully retrieved user workflows",
                response=WorkflowSerializer(many=True),
            ),
        },
        description="Get all workflows for the currently logged-in user",
    )
    @action(detail=False, methods=["get"])
    def my_workflows(self, request):
        """Get all workflows for the currently logged-in user."""
        queryset = self.get_queryset().order_by("-updated_at")

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

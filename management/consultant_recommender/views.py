"""Views for Consultant Recommender app."""

import logging
import time

from adrf import viewsets as async_viewsets
from asgiref.sync import sync_to_async
from django.db import models
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from authentication.models import CustomUser

from .models import (
    ConsultantRecommendation,
    ConsultantSearchLog,
    MeetingRequest,
    WorkflowConsultants,
)
from .serializers import (
    ConsultantRecommendationSerializer,
    ConsultantSearchLogSerializer,
    ConsultantSearchSerializer,
    MeetingBookingSerializer,
    MeetingRequestSerializer,
)
from .utils import (
    check_consultant_search_permission,
    consultant_cache,
    consultant_pinecone_service,
    decrement_user_consultant_search_count,
    get_user_consultant_search_quota,
)

logger = logging.getLogger(__name__)


@extend_schema_view(
    list=extend_schema(
        summary="List consultant search logs",
        description="Get a list of all consultant search logs for the current user",
        tags=["Consultant Recommender - Search"],
    ),
)
class ConsultantSearchViewSet(async_viewsets.ReadOnlyModelViewSet):
    """ViewSet for consultant searches using Pinecone + LLM (matches FastAPI)."""

    serializer_class = ConsultantSearchLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get search logs for current user."""
        if self.request.user.is_superuser:
            return ConsultantSearchLog.objects.all()
        return ConsultantSearchLog.objects.filter(user=self.request.user)

    @extend_schema(
        summary="Search for consultants by workflow",
        description="Search for consultants using workflow ID. Automatically fetches refined query from database and caches results.",
        parameters=[
            OpenApiParameter(
                name="workflow_id",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                required=True,
                description="Workflow ID to fetch refined query and search consultants",
            ),
        ],
        tags=["Consultant Recommender - Search"],
    )
    @action(detail=False, methods=["get"], url_path="search")
    async def search_consultants(self, request):
        """Search for consultants using Pinecone + LLM (FastAPI logic)."""
        serializer = ConsultantSearchSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)

        # Check search permission (sync operation)
        can_search, permission_msg, remaining_searches = await sync_to_async(
            check_consultant_search_permission
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

        start_time = time.time()

        try:
            workflow_id = serializer.validated_data["workflow_id"]
            logger.info(f"🔗 Fetching query data for workflow: {workflow_id}")

            # Fetch refined query and workflow data from database
            try:
                from ai_tool_recommender.models import RefinedQuery
                from workflow.models import Workflow

                refined_query_obj = await sync_to_async(
                    lambda: RefinedQuery.objects.filter(workflow_id=workflow_id).first()
                )()

                workflow_obj = await sync_to_async(
                    lambda: Workflow.objects.filter(
                        id=workflow_id, owner=request.user
                    ).first()
                )()

                if refined_query_obj:
                    query = refined_query_obj.refined_query
                    # Extract user work description from workflow_info if available
                    workflow_info = refined_query_obj.workflow_info or {}
                    user_work_description = workflow_info.get("outcome", "")
                    logger.info(f"✅ Found refined query: {query[:100]}...")
                elif workflow_obj and workflow_obj.user_query:
                    query = workflow_obj.user_query
                    user_work_description = workflow_obj.description or ""
                    logger.info(f"✅ Using workflow user_query: {query[:100]}...")
                else:
                    return Response(
                        {
                            "status": "error",
                            "error": "No query found for this workflow_id",
                            "message": "Workflow not found or no refined query available",
                            "workflow_id": workflow_id,
                        },
                        status=status.HTTP_404_NOT_FOUND,
                    )

            except Exception as e:
                logger.error(f"❌ Error fetching workflow data: {e}")
                return Response(
                    {
                        "status": "error",
                        "error": "Failed to fetch workflow data",
                        "message": str(e),
                        "workflow_id": workflow_id,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            logger.info(f"🔍 Consultant search - Query: {query}")
            logger.info(f"📝 User work description: {user_work_description}")
            logger.info(f"🔗 Workflow ID: {workflow_id}")

            # Check for workflow-specific cached consultants first if workflow_id is provided
            workflow_consultants = None
            if workflow_id:
                workflow_consultants = await sync_to_async(
                    lambda: WorkflowConsultants.objects.filter(
                        workflow_id=workflow_id, user=request.user
                    ).first()
                )()

                if workflow_consultants:
                    logger.info(
                        f"✨ Using workflow-cached consultant results for workflow {workflow_id}"
                    )
                    best_consultants = workflow_consultants.consultants_data
                    total_found = workflow_consultants.search_metadata.get(
                        "total_found", len(best_consultants)
                    )
                    response_time_ms = (time.time() - start_time) * 1000

                    # Return cached workflow consultants without decrementing search count
                    return Response(
                        {
                            "status": "success",
                            "workflow_id": workflow_id,
                            "consultants": best_consultants,
                            "consultants_count": len(best_consultants),
                            "cached": True,
                            "cached_at": workflow_consultants.updated_at.isoformat(),
                            "response_time_ms": response_time_ms,
                        }
                    )

            # Try to get cached results first
            cached_result = await consultant_cache.get(query, user_work_description)

            if cached_result:
                # Use cached results
                logger.info("✨ Using cached consultant search results")
                best_consultants = cached_result.get("best_consultants", [])
                total_found = cached_result.get("total_found", 0)
                response_time_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"✅ Returned {len(best_consultants)} cached consultants in {response_time_ms:.2f}ms"
                )
            else:
                # No cache - perform actual search
                logger.info("🔎 Cache miss - performing Pinecone + LLM search")

                # Use Pinecone + LLM to search consultants (FastAPI logic)
                search_result = await consultant_pinecone_service.search_consultants(
                    query=query,
                    user_work_description=user_work_description,
                    top_k=10,
                )

                response_time_ms = (time.time() - start_time) * 1000

                if search_result["status"] == "error":
                    raise Exception(search_result.get("error", "Search failed"))

                best_consultants = search_result.get("best_consultants", [])
                total_found = search_result.get("total_found", 0)

                logger.info(
                    f"✅ Found {total_found} consultants, filtered to {len(best_consultants)}"
                )

                # Cache the results for future requests
                await consultant_cache.set(
                    query,
                    user_work_description,
                    {
                        "best_consultants": best_consultants,
                        "total_found": total_found,
                    },
                )

            # Cache in workflow table if workflow_id is provided
            if workflow_id and best_consultants:
                search_metadata = {
                    "total_found": total_found,
                    "filtered_count": len(best_consultants),
                    "response_time_ms": response_time_ms,
                    "search_timestamp": time.time(),
                }

                # Create or update the workflow consultants cache
                if workflow_consultants:
                    # Update existing cache
                    workflow_consultants.query = query
                    workflow_consultants.user_work_description = user_work_description
                    workflow_consultants.consultants_data = best_consultants
                    workflow_consultants.search_metadata = search_metadata
                    await sync_to_async(workflow_consultants.save)()
                    logger.info(
                        f"✅ Updated workflow consultant cache for workflow {workflow_id}"
                    )
                else:
                    # Create new cache entry
                    await sync_to_async(WorkflowConsultants.objects.create)(
                        workflow_id=workflow_id,
                        user=request.user,
                        query=query,
                        user_work_description=user_work_description,
                        consultants_data=best_consultants,
                        search_metadata=search_metadata,
                    )
                    logger.info(
                        f"✅ Created new workflow consultant cache for workflow {workflow_id}"
                    )

            # Decrement user's search count (sync operation)
            success, decrement_msg = await sync_to_async(
                decrement_user_consultant_search_count
            )(request.user)

            if success:
                logger.info(f"Search count updated: {decrement_msg}")

            # Get updated remaining searches
            remaining_searches = await sync_to_async(get_user_consultant_search_quota)(
                request.user
            )

            # Log the search (async database operation)
            search_log = await sync_to_async(ConsultantSearchLog.objects.create)(
                user=request.user if request.user.is_authenticated else None,
                query=f"Workflow {workflow_id}: {query}",
                user_work_description=user_work_description,
                results_count=len(best_consultants),
                response_time_ms=response_time_ms,
                status="success" if best_consultants else "no_results",
            )
            logger.info(f"🔍 Search log created: {search_log}")
            # Return simplified response
            return Response(
                {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "consultants": best_consultants,
                    "consultants_count": len(best_consultants),
                    "cached": False,
                    "response_time_ms": response_time_ms,
                }
            )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in consultant search: {e}")

            # Log the error (async database operation)
            workflow_id = serializer.validated_data["workflow_id"]
            await sync_to_async(ConsultantSearchLog.objects.create)(
                user=request.user if request.user.is_authenticated else None,
                query=f"Workflow {workflow_id}: Error fetching consultants",
                user_work_description="",
                response_time_ms=response_time_ms,
                status="error",
                error_message=str(e),
            )

            return Response(
                {
                    "status": "error",
                    "error": str(e),
                    "workflow_id": workflow_id,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Get cached consultants for workflow",
        description="Retrieve cached consultants for a specific workflow without performing a new search.",
        parameters=[
            OpenApiParameter(
                name="workflow_id",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.QUERY,
                required=True,
                description="ID of the workflow to get cached consultants for",
            ),
        ],
        tags=["Consultant Recommender - Workflow"],
    )
    @action(detail=False, methods=["get"], url_path="get-workflow-consultants")
    async def get_workflow_consultants(self, request):
        """Get cached consultants for a specific workflow."""
        workflow_id = request.query_params.get("workflow_id")

        if not workflow_id:
            return Response(
                {
                    "status": "error",
                    "error": "workflow_id parameter is required",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get cached consultants for this workflow
            workflow_consultants = await sync_to_async(
                lambda: WorkflowConsultants.objects.filter(
                    workflow_id=workflow_id, user=request.user
                ).first()
            )()

            if not workflow_consultants:
                return Response(
                    {
                        "status": "not_found",
                        "message": "No cached consultants found for this workflow",
                        "workflow_id": workflow_id,
                        "consultants": [],
                        "consultants_count": 0,
                    }
                )

            return Response(
                {
                    "status": "success",
                    "workflow_id": workflow_id,
                    "consultants": workflow_consultants.consultants_data,
                    "consultants_count": workflow_consultants.consultants_count,
                    "query": workflow_consultants.query,
                    "user_work_description": workflow_consultants.user_work_description,
                    "search_metadata": workflow_consultants.search_metadata,
                    "cached_at": workflow_consultants.updated_at.isoformat(),
                    "created_at": workflow_consultants.created_at.isoformat(),
                }
            )

        except Exception as e:
            logger.error(f"❌ Error retrieving workflow consultants: {e}")
            return Response(
                {
                    "status": "error",
                    "error": str(e),
                    "workflow_id": workflow_id,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema_view(
    list=extend_schema(
        summary="List meeting requests",
        description="Get a list of meeting requests",
        tags=["Consultant Recommender - Meetings"],
    ),
    create=extend_schema(
        summary="Create meeting request",
        description="Create a new meeting request with a contractor",
        tags=["Consultant Recommender - Meetings"],
    ),
)
class MeetingRequestViewSet(viewsets.ModelViewSet):
    """ViewSet for meeting requests."""

    serializer_class = MeetingRequestSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get meeting requests for current user."""
        if self.request.user.is_superuser:
            return MeetingRequest.objects.all()

        # Users can see meetings they created or meetings with them as contractor
        return MeetingRequest.objects.filter(
            models.Q(client=self.request.user) | models.Q(contractor=self.request.user)
        ).order_by("-created_at")

    def perform_create(self, serializer):
        """Create meeting request."""
        serializer.save(client=self.request.user)

    @extend_schema(
        summary="Book a meeting",
        description="Book a meeting with a contractor",
        request=MeetingBookingSerializer,
        tags=["Consultant Recommender - Meetings"],
    )
    @action(detail=False, methods=["post"], url_path="book")
    def book_meeting(self, request):
        """Book a meeting with a contractor."""
        serializer = MeetingBookingSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            contractor_id = serializer.validated_data["contractor_id"]

            # Try to find contractor as CustomUser
            try:
                contractor = CustomUser.objects.get(
                    unique_id=contractor_id, role="contractor"
                )
                contractor_user = contractor
                contractor_id_external = None
                contractor_name = contractor.get_full_name() or contractor.username
                contractor_email = contractor.email
            except CustomUser.DoesNotExist:
                # External contractor
                contractor_user = None
                contractor_id_external = contractor_id
                contractor_name = serializer.validated_data.get(
                    "contractor_name", "Unknown"
                )
                contractor_email = serializer.validated_data.get("contractor_email", "")

            # Create meeting request
            meeting = MeetingRequest.objects.create(
                client=request.user,
                client_name=serializer.validated_data["client_name"],
                client_email=serializer.validated_data["client_email"],
                company_name=serializer.validated_data.get("company_name", ""),
                contractor=contractor_user,
                contractor_id_external=contractor_id_external,
                contractor_name=contractor_name,
                contractor_email=contractor_email,
                preferred_date=serializer.validated_data["preferred_date"],
                preferred_time=serializer.validated_data["preferred_time"],
                project_description=serializer.validated_data["project_description"],
                status="pending",
            )

            return Response(
                {
                    "status": "success",
                    "message": "Meeting request created successfully",
                    "meeting_id": str(meeting.id),
                    "meeting": MeetingRequestSerializer(meeting).data,
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

    @extend_schema(
        summary="Confirm a meeting",
        description="Confirm a meeting request (contractor only)",
        tags=["Consultant Recommender - Meetings"],
    )
    @action(detail=True, methods=["post"], permission_classes=[IsAuthenticated])
    def confirm(self, request, pk=None):
        """Confirm a meeting request."""
        meeting = self.get_object()

        # Only contractor can confirm
        if meeting.contractor != request.user:
            return Response(
                {"error": "Only the contractor can confirm this meeting"},
                status=status.HTTP_403_FORBIDDEN,
            )

        from django.utils import timezone

        meeting.status = "confirmed"
        meeting.confirmed_at = timezone.now()
        meeting.save()

        return Response(
            {
                "status": "success",
                "message": "Meeting confirmed",
                "meeting": MeetingRequestSerializer(meeting).data,
            }
        )

    @extend_schema(
        summary="Cancel a meeting",
        description="Cancel a meeting request",
        tags=["Consultant Recommender - Meetings"],
    )
    @action(detail=True, methods=["post"], permission_classes=[IsAuthenticated])
    def cancel(self, request, pk=None):
        """Cancel a meeting request."""
        meeting = self.get_object()

        # Either client or contractor can cancel
        if meeting.client != request.user and meeting.contractor != request.user:
            return Response(
                {"error": "You don't have permission to cancel this meeting"},
                status=status.HTTP_403_FORBIDDEN,
            )

        meeting.status = "cancelled"
        meeting.save()

        return Response(
            {
                "status": "success",
                "message": "Meeting cancelled",
                "meeting": MeetingRequestSerializer(meeting).data,
            }
        )


@extend_schema_view(
    list=extend_schema(
        summary="List consultant recommendations",
        description="Get a list of consultant recommendations",
        tags=["Consultant Recommender - Recommendations"],
    ),
)
class ConsultantRecommendationViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for consultant recommendations."""

    serializer_class = ConsultantRecommendationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Get recommendations for current user's searches."""
        if self.request.user.is_superuser:
            return ConsultantRecommendation.objects.all()

        return ConsultantRecommendation.objects.filter(
            search_log__user=self.request.user
        ).order_by("-created_at")

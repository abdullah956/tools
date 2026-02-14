"""Views for search functionality."""

import logging

from django.db.models import Avg, Count
from django.db.models.functions import TruncDate
from django.utils import timezone
from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .decorators import track_search_usage
from .models import SearchUsageLog
from .serializers import (
    DetailedSearchUsageLogSerializer,
    SearchQuerySerializer,
    SearchResultSerializer,
    SearchUsageLogSerializer,
    SearchUsageRangeDetailedSerializer,
    SearchUsageStatisticsSerializer,
)

logger = logging.getLogger(__name__)

# Get FastAPI service URL from settings


@extend_schema_view(
    create=extend_schema(
        description="Execute a search query",
        request=SearchQuerySerializer,
        responses={
            200: SearchResultSerializer,
            400: {"description": "Bad request"},
            401: {"description": "Authentication required"},
            403: {"description": "No active subscription"},
            429: {"description": "Search limit exceeded"},
        },
    )
)
class SearchViewSet(viewsets.ViewSet):
    """ViewSet for handling search requests."""

    permission_classes = [IsAuthenticated]
    serializer_class = SearchQuerySerializer

    @track_search_usage
    def create(self, request):
        """Handle POST requests to perform searches."""
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = request.user
            search_type = "trial" if user.trial_searches > 0 else "subscription"
            current_subscription = None

            # Import AI tool recommender to call directly instead of FastAPI
            from management.ai_tool_recommender.serializers import (
                SearchQuerySerializer as AISearchQuerySerializer,
            )
            from management.ai_tool_recommender.views import AIToolSearchViewSet

            # Create AI search request data
            ai_search_data = {
                "query": serializer.validated_data["query"],
                "max_results": 10,
                "include_pinecone": True,
                "include_internet": True,
            }

            # Add workflow_id if provided
            workflow_id = serializer.validated_data.get("workflow_id")
            if workflow_id:
                ai_search_data["workflow_id"] = workflow_id

            # Create AI search serializer and validate
            ai_serializer = AISearchQuerySerializer(data=ai_search_data)
            if not ai_serializer.is_valid():
                return Response(
                    ai_serializer.errors, status=status.HTTP_400_BAD_REQUEST
                )

            # Create AI tool search viewset and call search_tools method
            ai_viewset = AIToolSearchViewSet()
            ai_viewset.request = request  # Set the request object
            ai_viewset.format_kwarg = None  # Required for DRF viewsets

            # Call the search_tools method directly (it's async, so we need to handle that)
            import asyncio

            # Create a proper mock request with the validated data
            class MockRequest:
                def __init__(self, user, data):
                    self.user = user
                    self.data = data

            mock_request = MockRequest(user, ai_search_data)

            # Run the async search_tools method
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Call the search method directly
            search_response = loop.run_until_complete(
                ai_viewset.search_tools(mock_request)
            )

            # Extract the response data properly
            if hasattr(search_response, "data"):
                search_results = search_response.data
            else:
                # If it's already a dict (from Response object)
                search_results = search_response

            # Ensure search_results is a dict
            if not isinstance(search_results, dict):
                logger.error(f"Unexpected response type: {type(search_results)}")
                raise Exception("Invalid response format from AI search")

            # If using trial search, deduct from trial searches
            if search_type == "trial":
                user.trial_searches -= 1
                user.save()
            else:
                # Get current subscription for subscription searches
                current_subscription = user.subscriptions.filter(
                    status="active",
                    start_date__lte=timezone.now(),
                    end_date__gt=timezone.now(),
                ).first()

            # Create search log
            SearchUsageLog.objects.create(
                user=user,
                subscription=current_subscription,
                search_type=search_type,
                query=serializer.validated_data["query"],
                status="success",
            )

            # Calculate remaining searches
            subscription_searches_remaining = 0
            if current_subscription and search_type == "subscription":
                monthly_usage = SearchUsageLog.objects.filter(
                    user=user,
                    subscription=current_subscription,
                    search_type="subscription",
                    status="success",
                    timestamp__gte=current_subscription.start_date,
                    timestamp__lte=current_subscription.end_date,
                ).count()
                subscription_searches_remaining = (
                    current_subscription.product.searches_per_month - monthly_usage
                )

            # Add remaining searches info to response
            search_results["remaining_searches"] = {
                "trial": user.trial_searches,
                "subscription": subscription_searches_remaining,
            }

            return Response(search_results)

        except Exception as e:
            logger.error(f"Search error for user {request.user.id}: {str(e)}")
            return Response(
                {"error": "An error occurred while processing your search"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@extend_schema_view(
    list=extend_schema(
        description="List search usage logs for the current user",
        parameters=[
            OpenApiParameter(
                name="period",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Filter period (weekly/monthly/yearly/custom)",
                required=False,
                enum=["weekly", "monthly", "yearly", "custom"],
            ),
            OpenApiParameter(
                name="start_date",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Start date for custom period (YYYY-MM-DD)",
                required=False,
            ),
            OpenApiParameter(
                name="end_date",
                type=str,
                location=OpenApiParameter.QUERY,
                description="End date for custom period (YYYY-MM-DD)",
                required=False,
            ),
        ],
        responses={200: SearchUsageLogSerializer(many=True)},
    ),
    retrieve=extend_schema(
        description="Retrieve a specific search usage log",
        responses={200: SearchUsageLogSerializer},
    ),
)
class SearchUsageViewSet(
    mixins.ListModelMixin, mixins.RetrieveModelMixin, viewsets.GenericViewSet
):
    """ViewSet for managing search usage logs and statistics."""

    permission_classes = [IsAuthenticated]
    serializer_class = SearchUsageLogSerializer

    def get_queryset(self):
        """Return filtered search logs for the current user."""
        period = self.request.query_params.get("period", "weekly")
        start_date = self.request.query_params.get("start_date")
        end_date = self.request.query_params.get("end_date")

        queryset = SearchUsageLog.objects.filter(user=self.request.user)

        now = timezone.now()
        if period == "weekly":
            start = now - timezone.timedelta(days=7)
            queryset = queryset.filter(timestamp__gte=start)
        elif period == "monthly":
            start = now - timezone.timedelta(days=30)
            queryset = queryset.filter(timestamp__gte=start)
        elif period == "yearly":
            start = now - timezone.timedelta(days=365)
            queryset = queryset.filter(timestamp__gte=start)
        elif period == "custom" and start_date and end_date:
            try:
                start = timezone.datetime.strptime(start_date, "%Y-%m-%d")
                end = timezone.datetime.strptime(
                    end_date, "%Y-%m-%d"
                ) + timezone.timedelta(days=1)
                queryset = queryset.filter(timestamp__gte=start, timestamp__lt=end)
            except ValueError:
                pass

        return queryset.order_by("-timestamp")

    @extend_schema(
        description="Get search usage statistics for the current user",
        responses={200: SearchUsageStatisticsSerializer},
    )
    @action(detail=False, methods=["get"])
    def statistics(self, request):
        """Get search usage statistics for the current user."""
        try:
            stats = SearchUsageLog.get_usage_statistics(request.user)
            serializer = SearchUsageStatisticsSerializer(data=stats)
            if not serializer.is_valid():
                logger.error(f"Invalid statistics format: {serializer.errors}")
                raise ValueError("Invalid statistics format")
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Failed to retrieve search usage statistics: {e}")
            return Response(
                {"error": "Failed to retrieve search usage statistics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="period",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Filter period (weekly/monthly/yearly/custom)",
                required=True,
                enum=["weekly", "monthly", "yearly", "custom"],
            ),
            OpenApiParameter(
                name="start_date",
                type=str,
                location=OpenApiParameter.QUERY,
                description="Start date for custom period (YYYY-MM-DD)",
                required=False,
            ),
            OpenApiParameter(
                name="end_date",
                type=str,
                location=OpenApiParameter.QUERY,
                description="End date for custom period (YYYY-MM-DD)",
                required=False,
            ),
        ],
        responses={200: SearchUsageRangeDetailedSerializer},
    )
    @action(detail=False, methods=["get"], url_path="usage-range")
    def usage_range(self, request):
        """Get detailed search usage statistics for a specific period."""
        period = request.query_params.get("period", "weekly")
        start_date = request.query_params.get("start_date")
        end_date = request.query_params.get("end_date")

        now = timezone.now()

        # Determine date range based on period
        if period == "weekly":
            start_datetime = now - timezone.timedelta(days=7)
            end_datetime = now
        elif period == "monthly":
            start_datetime = now - timezone.timedelta(days=30)
            end_datetime = now
        elif period == "yearly":
            start_datetime = now - timezone.timedelta(days=365)
            end_datetime = now
        elif period == "custom":
            try:
                start_datetime = timezone.datetime.strptime(start_date, "%Y-%m-%d")
                end_datetime = timezone.datetime.strptime(
                    end_date, "%Y-%m-%d"
                ) + timezone.timedelta(days=1)
            except (ValueError, TypeError):
                return Response(
                    {"error": "Invalid date format for custom period. Use YYYY-MM-DD"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            return Response(
                {"error": "Invalid period specified"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get usage logs for the period
            usage_logs = SearchUsageLog.objects.filter(
                user=request.user,
                timestamp__gte=start_datetime,
                timestamp__lte=end_datetime,
            ).select_related("subscription", "subscription__product")

            # Calculate statistics
            status_counts = usage_logs.values("status", "search_type").annotate(
                count=Count("id")
            )

            # Initialize counters
            total_searches = usage_logs.count()
            successful_searches = 0
            failed_searches = 0
            limit_exceeded = 0
            trial_searches_used = 0
            subscription_searches_used = 0

            # Process status counts
            for item in status_counts:
                if item["status"] == "success":
                    successful_searches += item["count"]
                    if item["search_type"] == "trial":
                        trial_searches_used += item["count"]
                    else:
                        subscription_searches_used += item["count"]
                elif item["status"] == "error":
                    failed_searches += item["count"]
                elif item["status"] == "limit_exceeded":
                    limit_exceeded += item["count"]

            avg_response_time = (
                usage_logs.filter(response_time__isnull=False).aggregate(
                    avg_time=Avg("response_time")
                )["avg_time"]
                or 0
            )

            # Get current subscription details
            current_subscription = request.user.subscriptions.filter(
                status="active",
                start_date__lte=timezone.now(),
                end_date__gt=timezone.now(),
            ).first()

            subscription_limit = 0
            subscription_name = None
            subscription_type = None
            subscription_start = None
            subscription_end = None

            if current_subscription:
                subscription_limit = current_subscription.product.searches_per_month
                subscription_name = current_subscription.product.name
                subscription_type = current_subscription.product.name
                subscription_start = current_subscription.start_date
                subscription_end = current_subscription.end_date

            # Calculate usage percentage
            usage_percentage = (
                (subscription_searches_used / subscription_limit * 100)
                if subscription_limit > 0
                else 0
            )

            # Get daily usage summary with search type breakdown
            daily_usage = (
                usage_logs.annotate(date=TruncDate("timestamp"))
                .values("date", "status", "search_type")
                .annotate(count=Count("id"))
                .order_by("date")
            )

            # Format daily usage into a nested dictionary
            daily_usage_dict = {}
            for item in daily_usage:
                date_str = item["date"].strftime("%Y-%m-%d")
                if date_str not in daily_usage_dict:
                    daily_usage_dict[date_str] = {
                        "trial": {"success": 0, "error": 0, "limit_exceeded": 0},
                        "subscription": {"success": 0, "error": 0, "limit_exceeded": 0},
                    }
                search_type = item["search_type"] or "trial"  # Default to trial if None
                daily_usage_dict[date_str][search_type][item["status"]] = item["count"]

            # Get usage logs and serialize them
            DetailedSearchUsageLogSerializer(usage_logs, many=True).data

            response_data = {
                "start_date": start_datetime,
                "end_date": end_datetime,
                "total_searches": total_searches,
                "successful_searches": successful_searches,
                "failed_searches": failed_searches,
                "limit_exceeded_searches": limit_exceeded,
                "average_response_time": round(avg_response_time, 2),
                "subscription_limit": subscription_limit,
                "usage_percentage": round(usage_percentage, 2),
                "trial_searches_used": trial_searches_used,
                "trial_searches_remaining": request.user.trial_searches,
                "trial_searches_total": 10,
                "subscription_name": subscription_name,
                "subscription_type": subscription_type,
                "subscription_start": subscription_start,
                "subscription_end": subscription_end,
                "subscription_searches_used": subscription_searches_used,
                "daily_usage": daily_usage_dict,
            }

            # Remove usage_logs from response as requested
            serializer = SearchUsageRangeDetailedSerializer(data=response_data)
            if not serializer.is_valid():
                logger.error(f"Serializer errors: {serializer.errors}")
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error processing usage range request: {str(e)}")
            return Response(
                {"error": f"Error processing request: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

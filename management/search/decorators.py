"""Decorators for search functionality."""

import logging
import time
from functools import wraps

from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response

from subscription.models import UserSubscription

from .models import SearchUsageLog

logger = logging.getLogger(__name__)


def track_search_usage(view_func):
    """Decorator to track search API usage and enforce subscription limits.

    This decorator:
    1. Checks if user has trial searches remaining
    2. If no trial searches, checks for active subscription
    3. Logs the search attempt and its outcome
    4. Tracks response time
    """

    @wraps(view_func)
    def wrapper(view_instance, request, *args, **kwargs):
        start_time = time.time()
        user = request.user

        # Check authentication
        if not user.is_authenticated:
            return Response(
                {"error": "Authentication required"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        # First check if user has trial searches
        if user.trial_searches > 0:
            try:
                # Execute the view function for trial search
                response = view_func(view_instance, request, *args, **kwargs)
                return response
            except Exception as e:
                logger.error(f"Error in trial search for user {user.id}: {str(e)}")
                raise

        # If no trial searches left, check subscription
        current_subscription = (
            UserSubscription.objects.filter(
                user=user,
                status="active",
                start_date__lte=timezone.now(),
                end_date__gt=timezone.now(),
            )
            .select_related("product")
            .first()
        )

        if not current_subscription:
            logger.warning(
                f"No trial searches or active subscription for user {user.id}"
            )
            return Response(
                {
                    "error": "No searches available",
                    "details": (
                        "Trial searches exhausted and no active subscription found"
                    ),
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # Check subscription usage limits
        monthly_usage = SearchUsageLog.objects.filter(
            user=user,
            subscription=current_subscription,
            search_type="subscription",
            status="success",
            timestamp__gte=current_subscription.start_date,
            timestamp__lte=current_subscription.end_date,
        ).count()

        allowed_searches = current_subscription.product.searches_per_month

        if monthly_usage >= allowed_searches:
            logger.warning(f"Search limit exceeded for user {user.id}")
            SearchUsageLog.objects.create(
                user=user,
                subscription=current_subscription,
                search_type="subscription",
                query=request.data.get("query", ""),
                status="limit_exceeded",
                response_time=time.time() - start_time,
            )
            return Response(
                {
                    "error": "Monthly search limit exceeded",
                    "current_usage": monthly_usage,
                    "limit": allowed_searches,
                    "subscription_ends": current_subscription.end_date,
                },
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        try:
            # Execute the view function
            response = view_func(view_instance, request, *args, **kwargs)
            return response

        except Exception as e:
            # Log error
            logger.error(f"Error in search operation for user {user.id}: {str(e)}")
            SearchUsageLog.objects.create(
                user=user,
                subscription=current_subscription,
                search_type="subscription",
                query=request.data.get("query", ""),
                status="error",
                response_time=time.time() - start_time,
            )
            raise

    return wrapper

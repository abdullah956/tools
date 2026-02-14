"""Utility functions for the early access form app."""

from functools import wraps

from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response

from management.subscription.models import UserSubscription

from .models import SearchUsageLog


def track_search_usage():
    """Decorator to track search API usage and enforce subscription limits."""

    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(view_instance, request, *args, **kwargs):
            # Get the user
            user = request.user
            if not user.is_authenticated:
                return Response(
                    {"error": "Authentication required"},
                    status=status.HTTP_401_UNAUTHORIZED,
                )

            # Get current active subscription
            current_subscription = UserSubscription.objects.filter(
                user=user,
                status="active",
                start_date__lte=timezone.now(),
                end_date__gt=timezone.now(),
            ).first()

            # If no active subscription, user can't perform searches
            if not current_subscription:
                return Response(
                    {"error": "Active subscription required to perform searches"},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # Check search limits
            current_usage = SearchUsageLog.get_monthly_usage(user, current_subscription)
            allowed_searches = current_subscription.product.searches_per_month

            if current_usage >= allowed_searches:
                # Log the limit exceeded attempt
                SearchUsageLog.objects.create(
                    user=user,
                    subscription=current_subscription,
                    query=request.data.get("query", ""),
                    status="limit_exceeded",
                )
                return Response(
                    {
                        "error": "Monthly search limit exceeded",
                        "current_usage": current_usage,
                        "limit": allowed_searches,
                        "subscription_ends": current_subscription.end_date,
                    },
                    status=status.HTTP_429_TOO_MANY_REQUESTS,
                )

            # Execute the search
            response = view_func(view_instance, request, *args, **kwargs)

            # Log the successful search
            SearchUsageLog.objects.create(
                user=user,
                subscription=current_subscription,
                query=request.data.get("query", ""),
                status="success",
            )

            return response

        return wrapped_view

    return decorator

"""Utility functions for Consultant Recommender search quota management."""

import logging
from typing import Dict, Tuple

from django.utils import timezone

logger = logging.getLogger(__name__)


def get_user_consultant_search_quota(user) -> Dict[str, int]:
    """Get the user's remaining consultant search quota for trial and subscription.

    Args:
        user: CustomUser instance

    Returns:
        Dictionary with trial and subscription remaining searches
    """
    from subscription.models import UserSubscription

    # Get trial searches
    trial_searches = user.trial_searches if user and user.is_authenticated else 99999

    # Get subscription searches
    subscription_searches = 0
    if user and user.is_authenticated:
        # Get active subscription
        active_subscription = (
            UserSubscription.objects.filter(
                user=user, status="active", end_date__gte=timezone.now()
            )
            .select_related("product")
            .first()
        )

        if active_subscription:
            # For subscriptions, we return the monthly limit
            # In a real system, you'd track monthly usage
            subscription_searches = active_subscription.product.searches_per_month

    return {"trial": trial_searches, "subscription": subscription_searches}


def decrement_user_consultant_search_count(user) -> Tuple[bool, str]:
    """Decrement the user's consultant search count based on their subscription status.

    Args:
        user: CustomUser instance

    Returns:
        Tuple of (success: bool, message: str)
    """
    from subscription.models import UserSubscription

    if not user or not user.is_authenticated:
        # Unauthenticated users get unlimited searches (for demo)
        return True, "Unauthenticated user - unlimited searches"

    # Check if user has active subscription
    active_subscription = (
        UserSubscription.objects.filter(
            user=user, status="active", end_date__gte=timezone.now()
        )
        .select_related("product")
        .first()
    )

    if active_subscription:
        # User has active subscription - deduct from subscription searches
        # In a real system, you'd track this per month/period
        logger.info(
            f"User {user.email} has active subscription - allowing consultant search"
        )
        return True, "Active subscription"

    # No active subscription - use trial searches
    if user.trial_searches > 0:
        user.trial_searches -= 1
        user.save(update_fields=["trial_searches"])
        logger.info(
            f"Decremented trial consultant searches for {user.email}. "
            f"Remaining: {user.trial_searches}"
        )
        return True, f"Trial search used. Remaining: {user.trial_searches}"

    # No searches remaining
    logger.warning(f"User {user.email} has no remaining consultant searches")
    return False, "No remaining searches. Please subscribe to continue."


def check_consultant_search_permission(user) -> Tuple[bool, str, Dict[str, int]]:
    """Check if user has permission to perform a consultant search.

    Args:
        user: CustomUser instance

    Returns:
        Tuple of (can_search: bool, message: str, remaining: dict)
    """
    remaining = get_user_consultant_search_quota(user)

    if not user or not user.is_authenticated:
        # Demo mode - allow unlimited
        return True, "Demo mode", remaining

    # Check if user has any searches available
    if remaining["subscription"] > 0:
        return True, "Active subscription", remaining

    if remaining["trial"] > 0:
        return True, "Trial searches available", remaining

    return False, "No searches remaining", remaining

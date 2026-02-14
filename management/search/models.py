"""Models for search functionality."""

from datetime import datetime

from django.conf import settings
from django.db import models
from django.utils import timezone


class SearchUsageLog(models.Model):
    """Model to track search API usage."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="search_logs"
    )
    subscription = models.ForeignKey(
        "subscription.UserSubscription",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="search_logs",
    )
    search_type = models.CharField(
        max_length=20,
        choices=[
            ("trial", "Trial Search"),
            ("subscription", "Subscription Search"),
        ],
        default="trial",
        help_text="Whether this search used trial credits or subscription credits",
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    query = models.TextField(
        blank=True, null=True, help_text="The search query that was executed"
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("success", "Success"),
            ("error", "Error"),
            ("limit_exceeded", "Limit Exceeded"),
        ],
        default="success",
    )
    response_time = models.FloatField(null=True, blank=True)

    class Meta:
        """Meta class for SearchUsageLog."""

        indexes = [
            models.Index(fields=["user", "timestamp"]),
            models.Index(fields=["status", "timestamp"]),
        ]
        ordering = ["-timestamp"]
        verbose_name = "Search Usage Log"
        verbose_name_plural = "Search Usage Logs"

    @classmethod
    def get_monthly_usage(cls, user, current_subscription):
        """Get the number of successful searches."""
        if not current_subscription:
            return 0
        return cls.objects.filter(
            user=user,
            timestamp__gte=current_subscription.start_date,
            timestamp__lte=current_subscription.end_date,
            status="success",
        ).count()

    @classmethod
    def get_usage_statistics(cls, user):
        """Get comprehensive search usage statistics for a user with detailed metrics.

        Returns:
            dict: Dictionary containing comprehensive statistics for frontend graphs
        """
        from django.db.models import Avg, Count, Max, Min
        from django.db.models.functions import TruncDate

        try:
            # Get current subscription
            current_subscription = (
                user.subscriptions.filter(
                    status="active",
                    start_date__lte=timezone.now(),
                    end_date__gt=timezone.now(),
                )
                .select_related("product")
                .first()
            )

            # Calculate date ranges
            now = timezone.now()
            today = now.date()
            week_ago = now - timezone.timedelta(days=7)
            month_ago = now - timezone.timedelta(days=30)
            year_ago = now - timezone.timedelta(days=365)

            # Get all searches for the user
            all_searches = cls.objects.filter(user=user)

            # Trial search statistics
            trial_searches_used = all_searches.filter(
                search_type="trial", status="success"
            ).count()
            trial_searches_failed = all_searches.filter(
                search_type="trial", status="error"
            ).count()

            # Subscription search statistics
            subscription_searches_used = 0
            subscription_searches_failed = 0
            allowed_searches = 0
            subscription_start = None
            subscription_end = None
            subscription_name = None
            subscription_type = None
            days_remaining = None
            usage_percentage = 0

            if current_subscription:
                subscription_searches = all_searches.filter(
                    subscription=current_subscription,
                    search_type="subscription",
                    timestamp__gte=current_subscription.start_date,
                    timestamp__lte=current_subscription.end_date,
                )
                subscription_searches_used = subscription_searches.filter(
                    status="success"
                ).count()
                subscription_searches_failed = subscription_searches.filter(
                    status="error"
                ).count()

                allowed_searches = current_subscription.product.searches_per_month
                subscription_start = current_subscription.start_date
                subscription_end = current_subscription.end_date
                subscription_name = current_subscription.product.name
                subscription_type = current_subscription.product.product_type

                # Calculate days remaining for subscription
                days_remaining = max(0, (subscription_end - now).days)
                usage_percentage = (
                    (subscription_searches_used / allowed_searches * 100)
                    if allowed_searches > 0
                    else 0
                )
            else:
                # No active subscription - calculate trial days remaining if applicable
                # Trial typically lasts 30 days from account creation
                allowed_searches = user.trial_searches_total or 10
                if user.date_joined:
                    trial_end_date = user.date_joined + timezone.timedelta(days=30)
                    days_remaining = max(0, (trial_end_date - now).days)
                    # If trial expired, set to None
                    if days_remaining <= 0:
                        days_remaining = None
                else:
                    days_remaining = None

            # Time-based statistics
            # Use proper datetime range for today
            today_start = timezone.make_aware(
                datetime.combine(today, datetime.min.time())
            )
            today_end = today_start + timezone.timedelta(days=1)

            searches_today = all_searches.filter(
                timestamp__gte=today_start, timestamp__lt=today_end
            ).count()
            searches_this_week = all_searches.filter(timestamp__gte=week_ago).count()
            searches_this_month = all_searches.filter(timestamp__gte=month_ago).count()
            searches_this_year = all_searches.filter(timestamp__gte=year_ago).count()

            # Success/failure statistics
            total_successful = all_searches.filter(status="success").count()
            # Include all failed searches (both trial and subscription)
            total_failed = all_searches.filter(
                status__in=["error", "limit_exceeded"]
            ).count()
            total_limit_exceeded = all_searches.filter(status="limit_exceeded").count()
            success_rate = (
                (total_successful / (total_successful + total_failed) * 100)
                if (total_successful + total_failed) > 0
                else 0
            )

            # Performance statistics
            avg_response_time = (
                all_searches.filter(response_time__isnull=False).aggregate(
                    avg=Avg("response_time")
                )["avg"]
                or 0
            )
            min_response_time = (
                all_searches.filter(response_time__isnull=False).aggregate(
                    min=Min("response_time")
                )["min"]
                or 0
            )
            max_response_time = (
                all_searches.filter(response_time__isnull=False).aggregate(
                    max=Max("response_time")
                )["max"]
                or 0
            )

            # Daily usage for last 30 days (for graphs)
            daily_usage = (
                all_searches.filter(timestamp__gte=month_ago, status="success")
                .annotate(date=TruncDate("timestamp"))
                .values("date")
                .annotate(count=Count("id"))
                .order_by("date")
            )
            daily_usage_dict = {
                str(item["date"]): item["count"] for item in daily_usage
            }

            # Weekly usage for last 12 weeks (for graphs)
            weeks_ago_12 = now - timezone.timedelta(weeks=12)
            weekly_usage = []
            for i in range(12):
                week_start = weeks_ago_12 + timezone.timedelta(weeks=i)
                week_end = week_start + timezone.timedelta(weeks=1)
                week_count = all_searches.filter(
                    timestamp__gte=week_start,
                    timestamp__lt=week_end,
                    status="success",
                ).count()
                weekly_usage.append(
                    {
                        "week": f"Week {i + 1}",
                        "start_date": str(week_start.date()),
                        "count": week_count,
                    }
                )

            # Monthly usage for last 12 months (for graphs)
            monthly_usage = []
            for i in range(12):
                month_start = now - timezone.timedelta(days=30 * (11 - i))
                month_end = month_start + timezone.timedelta(days=30)
                month_count = all_searches.filter(
                    timestamp__gte=month_start,
                    timestamp__lt=month_end,
                    status="success",
                ).count()
                monthly_usage.append(
                    {
                        "month": month_start.strftime("%B %Y"),
                        "count": month_count,
                    }
                )

            # Peak usage analysis
            peak_day = (
                all_searches.filter(status="success")
                .annotate(date=TruncDate("timestamp"))
                .values("date")
                .annotate(count=Count("id"))
                .order_by("-count")
                .first()
            )
            peak_usage_day = peak_day["date"] if peak_day else None
            peak_usage_count = peak_day["count"] if peak_day else 0

            # First and last search dates
            first_search = all_searches.order_by("timestamp").first()
            last_search = all_searches.order_by("-timestamp").first()
            first_search_date = first_search.timestamp if first_search else None
            last_search_date = last_search.timestamp if last_search else None

            # Calculate remaining searches
            # If no subscription, show trial remaining; otherwise show subscription remaining
            if current_subscription:
                remaining_searches = max(
                    0, allowed_searches - subscription_searches_used
                )
            else:
                # No subscription - show trial remaining
                remaining_searches = user.trial_searches

            total_remaining = user.trial_searches + (
                remaining_searches if current_subscription else 0
            )

            return {
                # Overview statistics
                "total_searches": total_successful,
                "total_searches_all_time": all_searches.count(),
                "allowed_searches": allowed_searches,
                "remaining_searches": remaining_searches,
                "total_remaining": total_remaining,
                # Subscription details
                "has_active_subscription": bool(current_subscription),
                "subscription_name": subscription_name,
                "subscription_type": subscription_type,
                "subscription_start": subscription_start,
                "subscription_end": subscription_end,
                "days_remaining": days_remaining,
                "usage_percentage": round(usage_percentage, 2),
                # Trial statistics
                "trial_searches_remaining": user.trial_searches,
                "trial_searches_used": trial_searches_used,
                "trial_searches_failed": trial_searches_failed,
                "trial_searches_total": user.trial_searches_total or 10,
                # Subscription usage
                "subscription_searches_used": subscription_searches_used,
                "subscription_searches_failed": subscription_searches_failed,
                # Time-based statistics
                "searches_today": searches_today,
                "searches_this_week": searches_this_week,
                "searches_this_month": searches_this_month,
                "searches_this_year": searches_this_year,
                # Success/failure statistics
                "total_successful": total_successful,
                "total_failed": total_failed,
                "total_limit_exceeded": total_limit_exceeded,
                "success_rate": round(success_rate, 2),
                # Performance statistics
                "avg_response_time": round(avg_response_time, 2),
                "min_response_time": round(min_response_time, 2),
                "max_response_time": round(max_response_time, 2),
                # Historical data for graphs
                "daily_usage_last_30_days": daily_usage_dict,
                "weekly_usage_last_12_weeks": weekly_usage,
                "monthly_usage_last_12_months": monthly_usage,
                # Peak usage
                "peak_usage_day": str(peak_usage_day) if peak_usage_day else None,
                "peak_usage_count": peak_usage_count,
                # Activity timeline
                "first_search_date": first_search_date,
                "last_search_date": last_search_date,
                "account_age_days": (now - user.date_joined).days
                if user.date_joined
                else 0,
                # Status
                "error": "No active subscription found"
                if not current_subscription
                else None,
            }

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {
                "total_searches": 0,
                "total_searches_all_time": 0,
                "allowed_searches": 0,
                "remaining_searches": 0,
                "total_remaining": user.trial_searches,
                "has_active_subscription": False,
                "subscription_name": None,
                "subscription_type": None,
                "subscription_start": None,
                "subscription_end": None,
                "days_remaining": None,
                "usage_percentage": 0,
                "trial_searches_remaining": user.trial_searches,
                "trial_searches_used": 0,
                "trial_searches_failed": 0,
                "trial_searches_total": 10,
                "subscription_searches_used": 0,
                "subscription_searches_failed": 0,
                "searches_today": 0,
                "searches_this_week": 0,
                "searches_this_month": 0,
                "searches_this_year": 0,
                "total_successful": 0,
                "total_failed": 0,
                "total_limit_exceeded": 0,
                "success_rate": 0,
                "avg_response_time": 0,
                "min_response_time": 0,
                "max_response_time": 0,
                "daily_usage_last_30_days": {},
                "weekly_usage_last_12_weeks": [],
                "monthly_usage_last_12_months": [],
                "peak_usage_day": None,
                "peak_usage_count": 0,
                "first_search_date": None,
                "last_search_date": None,
                "account_age_days": 0,
                "error": str(e),
            }

    def __str__(self):
        """Return string representation of the search usage log."""
        return f"{self.user.email} - {self.status} - {self.timestamp}"

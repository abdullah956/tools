"""Serializers for search functionality."""

from rest_framework import serializers

from .models import SearchUsageLog


class SearchQuerySerializer(serializers.Serializer):
    """Serializer for search queries."""

    query = serializers.CharField(
        required=True, help_text="The search query to execute"
    )
    workflow_id = serializers.UUIDField(
        required=False,
        help_text="Optional workflow ID to update existing workflow instead of creating new one",
    )


class SearchResultSerializer(serializers.Serializer):
    """Serializer for search results."""

    user = serializers.DictField(help_text="User information")
    results = serializers.ListField(help_text="List of search results")
    query = serializers.CharField(help_text="The original search query")


class SearchUsageLogSerializer(serializers.ModelSerializer):
    """Serializer for search usage logs."""

    subscription_name = serializers.CharField(
        source="subscription.product.name", read_only=True, allow_null=True
    )
    search_type = serializers.CharField(read_only=True)

    class Meta:
        """Meta class for SearchUsageLogSerializer."""

        model = SearchUsageLog
        fields = [
            "id",
            "user",
            "subscription",
            "subscription_name",
            "search_type",
            "timestamp",
            "query",
            "status",
            "response_time",
        ]
        read_only_fields = fields


class SearchUsageStatisticsSerializer(serializers.Serializer):
    """Serializer for comprehensive search usage statistics."""

    # Overview statistics
    total_searches = serializers.IntegerField()
    total_searches_all_time = serializers.IntegerField()
    allowed_searches = serializers.IntegerField()
    remaining_searches = serializers.IntegerField()
    total_remaining = serializers.IntegerField()

    # Subscription details
    has_active_subscription = serializers.BooleanField()
    subscription_name = serializers.CharField(allow_null=True)
    subscription_type = serializers.CharField(allow_null=True)
    subscription_start = serializers.DateTimeField(allow_null=True)
    subscription_end = serializers.DateTimeField(allow_null=True)
    days_remaining = serializers.IntegerField(allow_null=True)
    usage_percentage = serializers.FloatField()

    # Trial statistics
    trial_searches_remaining = serializers.IntegerField()
    trial_searches_used = serializers.IntegerField()
    trial_searches_failed = serializers.IntegerField()
    trial_searches_total = serializers.IntegerField()

    # Subscription usage
    subscription_searches_used = serializers.IntegerField()
    subscription_searches_failed = serializers.IntegerField()

    # Time-based statistics
    searches_today = serializers.IntegerField()
    searches_this_week = serializers.IntegerField()
    searches_this_month = serializers.IntegerField()
    searches_this_year = serializers.IntegerField()

    # Success/failure statistics
    total_successful = serializers.IntegerField()
    total_failed = serializers.IntegerField()
    total_limit_exceeded = serializers.IntegerField()
    success_rate = serializers.FloatField()

    # Performance statistics
    avg_response_time = serializers.FloatField()
    min_response_time = serializers.FloatField()
    max_response_time = serializers.FloatField()

    # Historical data for graphs
    daily_usage_last_30_days = serializers.DictField()
    weekly_usage_last_12_weeks = serializers.ListField()
    monthly_usage_last_12_months = serializers.ListField()

    # Peak usage
    peak_usage_day = serializers.CharField(allow_null=True)
    peak_usage_count = serializers.IntegerField()

    # Activity timeline
    first_search_date = serializers.DateTimeField(allow_null=True)
    last_search_date = serializers.DateTimeField(allow_null=True)
    account_age_days = serializers.IntegerField()

    # Status
    error = serializers.CharField(allow_null=True)


class SearchUsageRangeStatisticsSerializer(serializers.Serializer):
    """Serializer for search usage statistics within a date range."""

    start_date = serializers.DateTimeField()
    end_date = serializers.DateTimeField()
    total_searches = serializers.IntegerField()
    successful_searches = serializers.IntegerField()
    failed_searches = serializers.IntegerField()
    limit_exceeded_searches = serializers.IntegerField()
    average_response_time = serializers.FloatField()
    subscription_limit = serializers.IntegerField()
    usage_percentage = serializers.FloatField()
    trial_searches_used = serializers.IntegerField()
    trial_searches_remaining = serializers.IntegerField()
    subscription_searches_used = serializers.IntegerField()


class DetailedSearchUsageLogSerializer(serializers.ModelSerializer):
    """Detailed serializer for search usage logs."""

    subscription_name = serializers.CharField(
        source="subscription.product.name", read_only=True, allow_null=True
    )
    subscription_type = serializers.CharField(
        source="subscription.product.name", read_only=True, allow_null=True
    )
    search_type = serializers.CharField(read_only=True)
    formatted_timestamp = serializers.SerializerMethodField()

    class Meta:
        """Meta class for DetailedSearchUsageLogSerializer."""

        model = SearchUsageLog
        fields = [
            "id",
            "subscription_name",
            "subscription_type",
            "search_type",
            "timestamp",
            "formatted_timestamp",
            "query",
            "status",
            "response_time",
        ]
        read_only_fields = fields

    def get_formatted_timestamp(self, obj):
        """Return formatted timestamp for frontend display."""
        return obj.timestamp.strftime("%Y-%m-%d %H:%M:%S") if obj.timestamp else None


class SearchUsageRangeDetailedSerializer(serializers.Serializer):
    """Combined serializer for statistics and usage logs."""

    # Statistics
    start_date = serializers.DateTimeField()
    end_date = serializers.DateTimeField()
    total_searches = serializers.IntegerField()
    successful_searches = serializers.IntegerField()
    failed_searches = serializers.IntegerField()
    limit_exceeded_searches = serializers.IntegerField()
    average_response_time = serializers.FloatField()
    subscription_limit = serializers.IntegerField()
    usage_percentage = serializers.FloatField()

    # Trial search statistics
    trial_searches_used = serializers.IntegerField()
    trial_searches_remaining = serializers.IntegerField()
    trial_searches_total = serializers.IntegerField(default=10)

    # Subscription details
    subscription_name = serializers.CharField(allow_null=True)
    subscription_type = serializers.CharField(allow_null=True)
    subscription_start = serializers.DateTimeField(allow_null=True)
    subscription_end = serializers.DateTimeField(allow_null=True)
    subscription_searches_used = serializers.IntegerField()

    # Daily usage summary
    daily_usage = serializers.DictField(
        child=serializers.DictField(
            child=serializers.DictField(child=serializers.IntegerField())
        )
    )

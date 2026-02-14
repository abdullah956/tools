"""Admin for the search API."""

from django.contrib import admin

from .models import SearchUsageLog


@admin.register(SearchUsageLog)
class SearchUsageLogAdmin(admin.ModelAdmin):
    """Admin interface for SearchUsageLog model."""

    list_display = [
        "id",
        "user",
        "search_type",
        "status",
        "timestamp",
        "query_preview",
    ]
    list_filter = [
        "status",
        "search_type",
        "timestamp",
    ]
    search_fields = [
        "user__email",
        "query",
    ]
    readonly_fields = [
        "id",
        "user",
        "subscription",
        "search_type",
        "timestamp",
        "query",
        "status",
        "response_time",
    ]
    date_hierarchy = "timestamp"
    ordering = ["-timestamp"]

    def query_preview(self, obj):
        """Display query preview."""
        if obj.query:
            return obj.query[:100] + "..." if len(obj.query) > 100 else obj.query
        return "N/A"

    query_preview.short_description = "Query"

"""Admin configuration for AI Tool Recommender app."""

from django.contrib import admin

from .models import (
    AIToolSearchLog,
    BackgroundTask,
    DiscoveredTool,
    RefinedQuery,
    ToolComparison,
    WorkflowGeneration,
)


@admin.register(AIToolSearchLog)
class AIToolSearchLogAdmin(admin.ModelAdmin):
    """Admin for AI Tool Search Logs."""

    list_display = (
        "id",
        "user",
        "query_preview",
        "total_results_count",
        "status",
        "response_time_ms",
        "cache_hit",
        "created_at",
    )
    list_filter = ("status", "cache_hit", "created_at")
    search_fields = ("query", "refined_query", "user__email")
    readonly_fields = (
        "id",
        "created_at",
        "response_time_ms",
        "pinecone_results_count",
        "internet_results_count",
        "total_results_count",
    )
    ordering = ("-created_at",)

    def query_preview(self, obj):
        """Show preview of query."""
        return obj.query[:50] + "..." if len(obj.query) > 50 else obj.query

    query_preview.short_description = "Query"


@admin.register(DiscoveredTool)
class DiscoveredToolAdmin(admin.ModelAdmin):
    """Admin for Discovered Tools."""

    list_display = (
        "title",
        "website",
        "source",
        "status",
        "relevance_score",
        "discovered_at",
    )
    list_filter = ("status", "source", "discovered_at")
    search_fields = ("title", "description", "website", "discovery_query")
    readonly_fields = ("id", "discovered_at", "reviewed_at", "added_at")
    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "title",
                    "description",
                    "category",
                    "features",
                    "tags",
                    "website",
                )
            },
        ),
        (
            "Social Media",
            {
                "fields": (
                    "twitter",
                    "facebook",
                    "linkedin",
                    "instagram",
                    "youtube",
                    "tiktok",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Pricing",
            {
                "fields": ("price_from", "price_to", "pricing_model"),
            },
        ),
        (
            "Discovery Metadata",
            {
                "fields": (
                    "source",
                    "discovery_query",
                    "relevance_score",
                    "status",
                    "tool",
                ),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("discovered_at", "reviewed_at", "added_at"),
                "classes": ("collapse",),
            },
        ),
    )
    ordering = ("-discovered_at",)


@admin.register(WorkflowGeneration)
class WorkflowGenerationAdmin(admin.ModelAdmin):
    """Admin for Workflow Generations."""

    list_display = (
        "id",
        "user",
        "query_preview",
        "tools_count",
        "generation_method",
        "generation_time_ms",
        "created_at",
    )
    list_filter = ("generation_method", "created_at")
    search_fields = ("query", "user__email")
    readonly_fields = (
        "id",
        "created_at",
        "generation_time_ms",
        "tools_count",
        "workflow_data",
    )
    filter_horizontal = ("tools",)
    ordering = ("-created_at",)

    def query_preview(self, obj):
        """Show preview of query."""
        return obj.query[:50] + "..." if len(obj.query) > 50 else obj.query

    query_preview.short_description = "Query"


@admin.register(BackgroundTask)
class BackgroundTaskAdmin(admin.ModelAdmin):
    """Admin for Background Tasks."""

    list_display = (
        "task_id",
        "task_type",
        "status",
        "duration_seconds",
        "created_at",
        "completed_at",
    )
    list_filter = ("status", "task_type", "created_at")
    search_fields = ("task_id", "error_message")
    readonly_fields = (
        "task_id",
        "created_at",
        "started_at",
        "completed_at",
        "duration_seconds",
    )
    ordering = ("-created_at",)


@admin.register(ToolComparison)
class ToolComparisonAdmin(admin.ModelAdmin):
    """Admin for Tool Comparisons."""

    list_display = (
        "id",
        "user",
        "workflow_preview",
        "original_tool_node_id",
        "original_tool_name",
        "total_alternatives_found",
        "search_time_ms",
        "created_at",
    )
    list_filter = ("created_at", "total_alternatives_found")
    search_fields = (
        "original_tool_node_id",
        "comparison_query",
        "user__email",
        "workflow__query",
    )
    readonly_fields = (
        "id",
        "created_at",
        "search_time_ms",
        "total_alternatives_found",
    )
    ordering = ("-created_at",)

    def workflow_preview(self, obj):
        """Show preview of workflow query."""
        if obj.workflow and obj.workflow.query:
            return (
                obj.workflow.query[:50] + "..."
                if len(obj.workflow.query) > 50
                else obj.workflow.query
            )
        return "No workflow"

    workflow_preview.short_description = "Workflow"

    def original_tool_name(self, obj):
        """Show original tool name."""
        return obj.original_tool_data.get("label", "Unknown Tool")

    original_tool_name.short_description = "Original Tool"


@admin.register(RefinedQuery)
class RefinedQueryAdmin(admin.ModelAdmin):
    """Admin for Refined Queries."""

    list_display = (
        "id",
        "workflow_id",
        "user",
        "query_preview",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("workflow_id", "original_query", "refined_query", "user__email")
    readonly_fields = ("id", "created_at", "updated_at", "workflow_info")
    ordering = ("-created_at",)

    def query_preview(self, obj):
        """Show preview of refined query."""
        return (
            obj.refined_query[:80] + "..."
            if len(obj.refined_query) > 80
            else obj.refined_query
        )

    query_preview.short_description = "Refined Query"

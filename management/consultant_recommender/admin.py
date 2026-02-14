"""Admin configuration for Consultant Recommender app."""

from django.contrib import admin

from .models import (
    ConsultantRecommendation,
    ConsultantSearchLog,
    MeetingRequest,
    WorkflowConsultants,
)


@admin.register(ConsultantSearchLog)
class ConsultantSearchLogAdmin(admin.ModelAdmin):
    """Admin for Consultant Search Logs."""

    list_display = (
        "id",
        "user",
        "query_preview",
        "results_count",
        "top_score",
        "status",
        "response_time_ms",
        "created_at",
    )
    list_filter = ("status", "created_at")
    search_fields = ("query", "user_work_description", "user__email")
    readonly_fields = (
        "id",
        "created_at",
        "response_time_ms",
        "results_count",
        "top_score",
    )
    ordering = ("-created_at",)

    def query_preview(self, obj):
        """Show preview of query."""
        return obj.query[:50] + "..." if len(obj.query) > 50 else obj.query

    query_preview.short_description = "Query"


@admin.register(MeetingRequest)
class MeetingRequestAdmin(admin.ModelAdmin):
    """Admin for Meeting Requests."""

    list_display = (
        "id",
        "client_name",
        "contractor_name",
        "preferred_date",
        "preferred_time",
        "status",
        "created_at",
    )
    list_filter = ("status", "preferred_date", "created_at")
    search_fields = (
        "client_name",
        "client_email",
        "contractor_name",
        "contractor_email",
        "company_name",
        "project_description",
    )
    readonly_fields = ("id", "created_at", "updated_at", "confirmed_at")
    fieldsets = (
        (
            "Client Information",
            {
                "fields": (
                    "client",
                    "client_name",
                    "client_email",
                    "company_name",
                )
            },
        ),
        (
            "Contractor Information",
            {
                "fields": (
                    "contractor",
                    "contractor_id_external",
                    "contractor_name",
                    "contractor_email",
                )
            },
        ),
        (
            "Meeting Details",
            {
                "fields": (
                    "preferred_date",
                    "preferred_time",
                    "project_description",
                    "status",
                )
            },
        ),
        (
            "Notes",
            {
                "fields": (
                    "client_notes",
                    "contractor_notes",
                    "admin_notes",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at", "confirmed_at"),
                "classes": ("collapse",),
            },
        ),
    )
    ordering = ("-created_at",)


@admin.register(ConsultantRecommendation)
class ConsultantRecommendationAdmin(admin.ModelAdmin):
    """Admin for Consultant Recommendations."""

    list_display = (
        "id",
        "get_consultant_name",
        "rank",
        "relevance_score",
        "viewed",
        "clicked",
        "contacted",
        "created_at",
    )
    list_filter = ("viewed", "clicked", "contacted", "created_at")
    search_fields = (
        "contractor_user__email",
        "contractor_user__first_name",
        "contractor_user__last_name",
        "consultant_external_id",
    )
    readonly_fields = (
        "id",
        "created_at",
        "viewed_at",
        "clicked_at",
        "contacted_at",
    )
    ordering = ("-created_at",)

    def get_consultant_name(self, obj):
        """Get consultant name."""
        if obj.contractor_user:
            return obj.contractor_user.get_full_name() or obj.contractor_user.email
        return obj.consultant_external_id

    get_consultant_name.short_description = "Consultant"


@admin.register(WorkflowConsultants)
class WorkflowConsultantsAdmin(admin.ModelAdmin):
    """Admin interface for WorkflowConsultants."""

    list_display = [
        "workflow_id",
        "user",
        "consultants_count",
        "query_preview",
        "created_at",
        "updated_at",
    ]
    list_filter = [
        "created_at",
        "updated_at",
        "consultants_count",
    ]
    search_fields = [
        "workflow_id",
        "user__email",
        "user__username",
        "query",
    ]
    readonly_fields = [
        "id",
        "consultants_count",
        "created_at",
        "updated_at",
    ]
    ordering = ["-updated_at"]

    fieldsets = (
        ("Workflow Information", {"fields": ("workflow_id", "user")}),
        ("Search Parameters", {"fields": ("query", "user_work_description")}),
        (
            "Consultant Data",
            {
                "fields": ("consultants_data", "consultants_count", "search_metadata"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def query_preview(self, obj):
        """Show a preview of the query."""
        return obj.query[:50] + "..." if len(obj.query) > 50 else obj.query

    query_preview.short_description = "Query Preview"

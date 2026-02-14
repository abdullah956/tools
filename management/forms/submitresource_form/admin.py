"""Admin configuration for the submit resource form app."""

from django.contrib import admin

from .models import SubmitResourceForm


@admin.register(SubmitResourceForm)
class SubmitResourceFormAdmin(admin.ModelAdmin):
    """Admin configuration for SubmitResourceForm model."""

    list_display = ("title", "url", "created_at", "updated_at")
    search_fields = ("title", "url", "description")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)

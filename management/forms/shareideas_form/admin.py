"""Admin configuration for the share ideas form app."""

from django.contrib import admin

from .models import ShareIdeasForm


@admin.register(ShareIdeasForm)
class ShareIdeasFormAdmin(admin.ModelAdmin):
    """Admin configuration for ShareIdeasForm model."""

    list_display = ("name", "created_at", "updated_at")
    search_fields = ("name", "feedback")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)

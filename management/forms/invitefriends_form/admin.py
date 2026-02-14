"""Admin configuration for the invite friends form app."""

from django.contrib import admin

from .models import InviteFriendsForm


@admin.register(InviteFriendsForm)
class InviteFriendsFormAdmin(admin.ModelAdmin):
    """Admin configuration for InviteFriendsForm model."""

    list_display = ("get_emails_display", "created_at", "updated_at")
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)

    def get_emails_display(self, obj):
        """Display emails in a readable format."""
        return ", ".join(obj.emails)

    get_emails_display.short_description = "Invited Emails"

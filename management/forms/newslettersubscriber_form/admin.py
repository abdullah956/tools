"""Admin configuration for the newsletter subscriber form app."""

from django.contrib import admin

from .models import NewsletterSubscriberForm


@admin.register(NewsletterSubscriberForm)
class NewsletterSubscriberFormAdmin(admin.ModelAdmin):
    """Admin configuration for NewsletterSubscriberForm model."""

    list_display = ("email", "created_at", "updated_at")
    search_fields = ("email",)
    readonly_fields = ("created_at", "updated_at")
    ordering = ("-created_at",)

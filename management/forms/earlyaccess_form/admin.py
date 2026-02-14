"""Admin configuration for the early access form app."""

from django.contrib import admin

from .models import EarlyAccessForm


@admin.register(EarlyAccessForm)
class EarlyAccessFormAdmin(admin.ModelAdmin):
    """Admin configuration for EarlyAccessForm model."""

    list_display = (
        "email",
        "payment_status",
        "is_email_verified",
        "has_paid",
        "payment_amount",
        "payment_date",
        "created_at",
    )
    list_filter = (
        "payment_status",
        "is_email_verified",
        "has_paid",
        "created_at",
    )
    search_fields = ("email", "stripe_payment_intent_id", "is_email_verified")
    readonly_fields = (
        "created_at",
        "updated_at",
        "stripe_checkout_session_id",
        "stripe_payment_intent_id",
        "is_email_verified",
    )
    ordering = ("-created_at",)

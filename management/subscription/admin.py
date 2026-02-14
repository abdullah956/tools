"""Admin configuration for subscription models."""

from django.contrib import admin

from .models import SubscriptionProduct, UserSubscription


@admin.register(SubscriptionProduct)
class SubscriptionProductAdmin(admin.ModelAdmin):
    """Admin configuration for SubscriptionProduct model."""

    list_display = (
        "name",
        "price",
        "duration_months",
        "searches_per_month",
        "active_projects",
        "stripe_product_id",
        "created_at",
    )
    list_filter = ("early_adopter_benefits", "community_access", "priority_support")
    search_fields = ("name", "description", "stripe_product_id")
    readonly_fields = (
        "stripe_product_id",
        "stripe_price_id",
        "created_at",
        "updated_at",
    )
    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "name",
                    "description",
                    "price",
                    "duration_months",
                    "regular_price",
                )
            },
        ),
        (
            "Features",
            {
                "fields": (
                    "searches_per_month",
                    "active_projects",
                    "early_adopter_benefits",
                    "community_access",
                    "priority_support",
                )
            },
        ),
        (
            "Stripe Information",
            {
                "fields": ("stripe_product_id", "stripe_price_id"),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(UserSubscription)
class UserSubscriptionAdmin(admin.ModelAdmin):
    """Admin configuration for UserSubscription model."""

    list_display = ("user", "product", "status", "start_date", "end_date", "created_at")
    list_filter = ("status", "start_date", "end_date")
    search_fields = (
        "user__email",
        "user__username",
        "stripe_checkout_session_id",
        "stripe_payment_intent_id",
    )
    readonly_fields = (
        "stripe_checkout_session_id",
        "stripe_payment_intent_id",
        "created_at",
        "updated_at",
    )
    fieldsets = (
        ("User Information", {"fields": ("user", "product", "status")}),
        ("Subscription Period", {"fields": ("start_date", "end_date")}),
        (
            "Stripe Information",
            {
                "fields": (
                    "stripe_checkout_session_id",
                    "stripe_payment_intent_id",
                    "referral_id",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )
    raw_id_fields = ("user", "product")

    def get_queryset(self, request):
        """Optimize queryset by prefetching related fields."""
        return super().get_queryset(request).select_related("user", "product")

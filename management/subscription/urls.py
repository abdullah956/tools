"""URL configuration for subscription app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

app_name = "subscription"

router = DefaultRouter()
router.register(
    r"subscription-products",
    views.SubscriptionProductViewSet,
    basename="subscription-product",
)

router.register(
    r"user-subscriptions",
    views.UserSubscriptionViewSet,
    basename="user-subscription",
)


urlpatterns = [
    # API endpoints
    path("api/", include(router.urls)),
    # Checkout success and cancel endpoints
    path("checkout/success/", views.checkout_success, name="checkout-success"),
    path("checkout/cancel/", views.checkout_cancel, name="checkout-cancel"),
    # Stripe webhook endpoint
    path("webhook/stripe/", views.stripe_webhook, name="stripe-webhook"),
]

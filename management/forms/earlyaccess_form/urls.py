"""URL configuration for the early access form app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    EarlyAccessFormViewSet,
    WebhookView,
    checkout_cancel,
    checkout_success,
)

app_name = "earlyaccess_form"

router = DefaultRouter()
router.register(r"forms", EarlyAccessFormViewSet, basename="early-access")

urlpatterns = [
    path("", include(router.urls)),
    path("success/", checkout_success, name="early-access-success"),
    path("cancel/", checkout_cancel, name="early-access-cancel"),
    path("webhook/", WebhookView.as_view(), name="early-access-webhook"),
]

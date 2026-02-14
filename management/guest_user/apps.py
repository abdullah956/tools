"""App configuration for the guest user app."""

from django.apps import AppConfig


class GuestUserConfig(AppConfig):
    """Configuration for the guest user app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "guest_user"

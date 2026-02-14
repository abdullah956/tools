"""CHAT APP."""
from django.apps import AppConfig


class ChatConfig(AppConfig):
    """CHAT APP CONFIG."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "management.chat"
    label = "chat"

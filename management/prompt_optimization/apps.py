"""App configuration for the prompt optimization app."""

from django.apps import AppConfig


class QueryRefinementConfig(AppConfig):
    """Configuration for the prompt optimization app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "prompt_optimization"

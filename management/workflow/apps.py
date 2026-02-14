"""App configuration for the workflow app."""

from django.apps import AppConfig


class WorkflowConfig(AppConfig):
    """Configuration for the workflow app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "workflow"

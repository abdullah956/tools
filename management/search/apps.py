"""App configuration for search functionality."""

from django.apps import AppConfig


class SearchConfig(AppConfig):
    """Configuration for the search app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "management.search"
    label = "search"

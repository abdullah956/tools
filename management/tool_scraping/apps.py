"""Tool scraping Django app configuration."""

from django.apps import AppConfig


class ToolScrapingConfig(AppConfig):
    """Configuration for the tool_scraping app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "management.tool_scraping"
    verbose_name = "Tool Scraping"

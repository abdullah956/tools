"""This module contains the AppConfig for the tools."""

from django.apps import AppConfig


class ToolsConfig(AppConfig):
    """This class contains the AppConfig for the tools."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "management.tools"
    label = "tools"

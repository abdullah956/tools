"""Contractor app configuration."""
from django.apps import AppConfig


class ContractorConfig(AppConfig):
    """Contractor app configuration."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "management.contractor"
    label = "contractor"

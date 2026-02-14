"""App configuration for Consultant Recommender."""

from django.apps import AppConfig


class ConsultantRecommenderConfig(AppConfig):
    """Configuration for Consultant Recommender app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "consultant_recommender"
    verbose_name = "Consultant Recommender"

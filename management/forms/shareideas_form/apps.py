"""Apps for the share ideas form app."""

from django.apps import AppConfig


class ShareideasFormConfig(AppConfig):
    """Configuration for the share ideas form app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "forms.shareideas_form"
    verbose_name = "Share Ideas Form"

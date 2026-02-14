"""Apps for the early access form app."""

from django.apps import AppConfig


class EarlyaccessFormConfig(AppConfig):
    """Configuration for the early access form app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "forms.earlyaccess_form"
    verbose_name = "Early Access Form"

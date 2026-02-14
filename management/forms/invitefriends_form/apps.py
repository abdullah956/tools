"""Apps for the invite friends form app."""

from django.apps import AppConfig


class InvitefriendsFormConfig(AppConfig):
    """Configuration for the invite friends form app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "forms.invitefriends_form"
    verbose_name = "Invite Friends Form"

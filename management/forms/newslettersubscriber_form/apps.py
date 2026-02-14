"""Apps for the newsletter subscriber form app."""

from django.apps import AppConfig


class NewslettersubscriberFormConfig(AppConfig):
    """Configuration for the newsletter subscriber form app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "forms.newslettersubscriber_form"
    verbose_name = "Newsletter Subscriber Form"

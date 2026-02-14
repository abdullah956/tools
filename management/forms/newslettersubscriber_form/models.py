"""Models for the newsletter subscriber form app."""

from django.db import models


class NewsletterSubscriberForm(models.Model):
    """Model for newsletter subscriber registration form."""

    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta class for NewsletterSubscriberForm."""

        verbose_name = "Newsletter Subscriber Form"
        verbose_name_plural = "Newsletter Subscriber Forms"
        ordering = ["-created_at"]

    def __str__(self):
        """Return string representation of the newsletter subscriber form."""
        return f"Newsletter Subscription - {self.email}"

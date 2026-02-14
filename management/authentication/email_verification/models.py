# models.py
"""This module defines the models for email verification tokens."""

from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone


class EmailVerificationToken(models.Model):
    """Model to store email verification tokens for users."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    token = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    def is_expired(self):
        """Check if the token has expired."""
        return timezone.now() > self.expires_at

    def save(self, *args, **kwargs):
        """Save the token and set the expiration date if not already set."""
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(
                days=1
            )  # Token valid for 1 day
        super().save(*args, **kwargs)

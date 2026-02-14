"""Models for the invite friends form app."""

from django.contrib.postgres.fields import ArrayField
from django.db import models


class InviteFriendsForm(models.Model):
    """Model for inviting friends form."""

    emails = ArrayField(
        models.EmailField(),
        verbose_name="Friend's Emails",
        help_text="Enter up to 3 email addresses of friends you'd like to invite",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta class for InviteFriendsForm."""

        verbose_name = "Invite Friends Form"
        verbose_name_plural = "Invite Friends Forms"
        ordering = ["-created_at"]

    def __str__(self):
        """Return string representation of the invite submission."""
        return f"Invitation to {', '.join(self.emails)}"

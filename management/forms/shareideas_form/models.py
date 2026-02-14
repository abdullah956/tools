"""Models for the share ideas form app."""

from django.db import models


class ShareIdeasForm(models.Model):
    """Model for ideas sharing form."""

    name = models.CharField(max_length=255, verbose_name="Name")
    feedback = models.TextField(verbose_name="Feedback/Ideas")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta class for ShareIdeasForm."""

        verbose_name = "Share Ideas Form"
        verbose_name_plural = "Share Ideas Forms"
        ordering = ["-created_at"]

    def __str__(self):
        """Return string representation of the ideas submission."""
        return f"Ideas from {self.name}"

"""Models for the submit resource form app."""

from django.db import models


class SubmitResourceForm(models.Model):
    """Model for resource submission form."""

    url = models.URLField(verbose_name="Resource URL")
    title = models.CharField(max_length=255, verbose_name="Resource Title")
    description = models.TextField(verbose_name="Resource Description")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta class for SubmitResourceForm."""

        verbose_name = "Submit Resource Form"
        verbose_name_plural = "Submit Resource Forms"
        ordering = ["-created_at"]

    def __str__(self):
        """Return string representation of the resource submission."""
        return f"Resource Submission - {self.title}"

"""Models for the contractor app."""
from django.db import models


class Consultant(models.Model):
    """Model for storing consultant information."""

    name = models.CharField(max_length=255, null=True, blank=True)
    expertise = models.TextField(null=True, blank=True)
    experience = models.TextField(null=True, blank=True)
    website = models.URLField(max_length=255, null=True, blank=True)
    phone = models.CharField(max_length=50, null=True, blank=True)
    gmail = models.EmailField(null=True, blank=True)
    apps_included = models.TextField(null=True, blank=True)
    language = models.CharField(max_length=100, null=True, blank=True)
    country = models.CharField(max_length=100, null=True, blank=True)
    company_name = models.CharField(max_length=255, null=True, blank=True)
    type_of_services = models.TextField(null=True, blank=True)
    countries_with_office_locations = models.TextField(null=True, blank=True)
    about = models.TextField(null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    time = models.TimeField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True)  # Store embeddings as JSON

    def __str__(self):
        """String representation of the consultant."""
        return self.company_name or "Unnamed Consultant"

    class Meta:
        """Meta class for the consultant model."""

        db_table = "consultants"

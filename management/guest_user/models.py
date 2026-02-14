"""Models for the guest user app."""

from django.db import models

# Create your models here.


class GuestUser(models.Model):
    """Model representing a guest user."""

    ip_address = models.GenericIPAddressField(null=True, blank=True)
    no_of_requests = models.IntegerField(default=0)
    refresh_token = models.CharField(max_length=800, null=True, blank=True)
    access_token = models.CharField(max_length=800, null=True, blank=True)
    workflow_count = models.IntegerField(default=0)

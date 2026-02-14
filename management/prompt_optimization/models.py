"""Models for the prompt optimization app."""

from django.conf import settings
from django.db import models

from workflow.models import Workflow


class UserQuery(models.Model):
    """Model representing a user query and its refined version."""

    original_query = models.TextField()
    refined_query = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    workflow = models.ForeignKey(
        Workflow, on_delete=models.CASCADE, null=True, blank=True
    )

    def __str__(self):
        """Return a string representation of the original query."""
        return self.original_query

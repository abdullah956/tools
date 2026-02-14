"""Models for the workflow app."""

import uuid

from django.conf import settings
from django.contrib.postgres.search import (
    Coalesce,
    SearchQuery,
    SearchRank,
    SearchVector,
    TrigramSimilarity,
)
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q


def validate_node_data(value):
    """Validate the data for a node."""
    required_keys = {"label": str, "type": str, "description": str, "config": dict}
    config_keys = {"inputs": list, "outputs": list}

    # Check for required keys in the main data
    for key, expected_type in required_keys.items():
        if key not in value or not isinstance(value[key], expected_type):
            raise ValidationError(
                f"'{key}' is required and must be of type {expected_type.__name__}"
            )

    # Check for required keys in the config
    config = value.get("config", {})
    for key, expected_type in config_keys.items():
        if key not in config or not isinstance(config[key], expected_type):
            raise ValidationError(
                f"'config.{key}' is required and type {expected_type.__name__}"
            )


def validate_metadata(value):
    """Validate the metadata for a workflow."""
    required_keys = {
        "author": str,
        "version": str,
        "tags": list,
        "ai_generated": bool,
        "prompt": str,
    }

    # Check for required keys in the metadata
    for key, expected_type in required_keys.items():
        if key not in value or not isinstance(value[key], expected_type):
            raise ValidationError(
                f"'{key}' is required and must be of type {expected_type.__name__}"
            )


# Create your models here.
class Workflow(models.Model):
    """Model representing a workflow."""

    id = models.CharField(
        primary_key=True, max_length=36, default=uuid.uuid4, editable=False
    )
    name = models.CharField(max_length=255, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=50, null=True, blank=True)
    metadata = models.JSONField(null=True, blank=True, validators=[validate_metadata])
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    prompt = models.TextField(null=True, blank=True)
    user_query = models.TextField(null=True, blank=True)

    def save(self, *args, **kwargs):
        """Save the workflow, setting a default name if not provided."""
        if not self.name:
            self.name = "Unnamed Workflow"

        super().save(*args, **kwargs)
        print(f"Workflow {self.id} saved.")

    @classmethod
    def search(cls, query):
        """Perform precise search."""
        if not query:
            return cls.objects.none()

        # Full-text search vector with weights
        search_vector = (
            SearchVector("name", weight="A")
            + SearchVector("description", weight="B")
            + SearchVector("user_query", weight="C")
            + SearchVector("prompt", weight="C")
        )
        search_query = SearchQuery(query)

        # Calculate similarity scores with coalesce to handle NULL values
        return (
            cls.objects.annotate(
                rank=SearchRank(search_vector, search_query),
                similarity_name=Coalesce(TrigramSimilarity("name", query), 0.0),
                similarity_desc=Coalesce(TrigramSimilarity("description", query), 0.0),
                similarity_query=Coalesce(TrigramSimilarity("user_query", query), 0.0),
                similarity_prompt=Coalesce(TrigramSimilarity("prompt", query), 0.0),
                total_similarity=Coalesce(
                    (
                        TrigramSimilarity("name", query) * 2.0
                        + TrigramSimilarity("description", query) * 1.5
                        + TrigramSimilarity("user_query", query) * 1.0
                        + TrigramSimilarity("prompt", query) * 1.0
                    ),
                    0.0,
                ),
            )
            .filter(
                Q(
                    Q(name__icontains=query)
                    | Q(description__icontains=query)
                    | Q(user_query__icontains=query)
                    | Q(prompt__icontains=query)
                )
                | Q(rank__gte=0.3)
                | Q(total_similarity__gte=0.3)
            )
            .order_by("-total_similarity", "-rank", "name")
            .distinct()
        )


class Node(models.Model):
    """Model representing a node in a workflow."""

    id = models.CharField(
        primary_key=True, max_length=36, default=uuid.uuid4, editable=False
    )
    workflow = models.ForeignKey(
        Workflow, related_name="nodes", on_delete=models.CASCADE, null=True, blank=True
    )
    type = models.CharField(max_length=50, null=True, blank=True)
    position_x = models.IntegerField(null=True, blank=True)
    position_y = models.IntegerField(null=True, blank=True)
    data = models.JSONField(null=True, blank=True, validators=[validate_node_data])
    source_handles = models.JSONField(null=True, blank=True)
    dragging = models.BooleanField(default=False)
    height = models.IntegerField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    position_absolute_x = models.IntegerField(null=True, blank=True)
    position_absolute_y = models.IntegerField(null=True, blank=True)
    selected = models.BooleanField(default=False)

    def save(self, *args, **kwargs):
        """Save the node, setting a default type if not provided."""
        if not self.type:
            self.type = "default"
        super().save(*args, **kwargs)
        print(f"Node {self.id} saved.")


class Edge(models.Model):
    """Model representing an edge in a workflow."""

    id = models.CharField(
        primary_key=True, max_length=36, default=uuid.uuid4, editable=False
    )
    workflow = models.ForeignKey(
        Workflow, related_name="edges", on_delete=models.CASCADE, null=True, blank=True
    )
    source = models.CharField(max_length=100, null=True, blank=True)
    target = models.CharField(max_length=100, null=True, blank=True)
    source_handle = models.CharField(max_length=50, null=True, blank=True)
    type = models.CharField(max_length=50, null=True, blank=True)

    def save(self, *args, **kwargs):
        """Save the edge, setting a default type if not provided."""
        if not self.type:
            self.type = "default"
        super().save(*args, **kwargs)
        print(f"Edge {self.id} saved.")

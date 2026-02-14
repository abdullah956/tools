"""This module contains the models for the tools."""

import logging
import uuid

import numpy as np
from django.contrib.postgres.search import (
    Coalesce,
    SearchQuery,
    SearchRank,
    SearchVector,
    TrigramSimilarity,
)
from django.db import models
from django.db.models import Q
from django.db.models.signals import post_save
from django.dispatch import receiver
from envs.env_loader import env_loader
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class Tool(models.Model):
    """This class contains the Tool model."""

    class Meta:
        """This class contains the Meta class for the Tool model."""

        app_label = "tools"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=100)
    features = models.TextField()
    tags = models.CharField(max_length=500)
    website = models.CharField(max_length=255, blank=True)
    twitter = models.CharField(max_length=255, blank=True)
    facebook = models.CharField(max_length=255, blank=True)
    linkedin = models.CharField(max_length=255, blank=True)
    tiktok = models.CharField(max_length=255, blank=True)
    youtube = models.CharField(max_length=255, blank=True)
    instagram = models.CharField(max_length=255, blank=True)
    price_from = models.CharField(max_length=50)
    price_to = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        """This method returns the title of the tool."""
        return self.title

    @classmethod
    def search(cls, query):
        """Perform precise search across tool fields."""
        if not query:
            return cls.objects.none()

        # Full-text search vector with weights
        search_vector = (
            SearchVector("title", weight="A")
            + SearchVector("description", weight="B")
            + SearchVector("category", weight="B")
            + SearchVector("features", weight="C")
            + SearchVector("tags", weight="C")
        )
        search_query = SearchQuery(query)

        # Calculate similarity scores with coalesce to handle NULL values
        return (
            cls.objects.annotate(
                rank=SearchRank(search_vector, search_query),
                similarity_title=Coalesce(TrigramSimilarity("title", query), 0.0),
                similarity_desc=Coalesce(TrigramSimilarity("description", query), 0.0),
                similarity_category=Coalesce(TrigramSimilarity("category", query), 0.0),
                similarity_features=Coalesce(TrigramSimilarity("features", query), 0.0),
                similarity_tags=Coalesce(TrigramSimilarity("tags", query), 0.0),
                total_similarity=Coalesce(
                    (
                        TrigramSimilarity("title", query) * 2.0
                        + TrigramSimilarity("description", query) * 1.5
                        + TrigramSimilarity("category", query) * 1.5
                        + TrigramSimilarity("features", query) * 1.0
                        + TrigramSimilarity("tags", query) * 1.0
                    ),
                    0.0,
                ),
            )
            .filter(
                Q(
                    Q(rank__gte=0.4)  # Increased threshold for rank
                    | Q(total_similarity__gte=0.4)  # Increased threshold for similarity
                    | Q(
                        Q(similarity_title__gte=0.3)  # Title-specific threshold
                        | Q(
                            Q(similarity_desc__gte=0.4)
                            & Q(similarity_category__gte=0.4)
                        )  # Combined field threshold
                    )
                )
            )
            .order_by("-total_similarity", "-rank", "title")
            .distinct()
        )


@receiver(post_save, sender=Tool)
def update_pinecone_index(sender, instance, created, **kwargs):
    """This method updates the Pinecone index."""
    try:
        # Initialize Pinecone client
        pinecone_client = Pinecone(
            api_key=env_loader.pinecone_api_key,
        )

        # Get or create the index
        index_name = env_loader.pinecone_tool_index
        indexes = pinecone_client.list_indexes()

        if index_name not in [idx.name for idx in indexes]:
            # Create index with proper spec
            pinecone_client.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        # Connect to the index
        index = pinecone_client.Index(index_name)

        # Create tool metadata
        metadata = {
            "id": str(instance.id),  # UUID will be automatically converted to string
            "title": instance.title,
            "description": instance.description,
            "category": instance.category,
            "features": instance.features,
            "tags": instance.tags,
            "website": instance.website,
            "twitter": instance.twitter,
            "facebook": instance.facebook,
            "linkedin": instance.linkedin,
            "tiktok": instance.tiktok,
            "youtube": instance.youtube,
            "instagram": instance.instagram,
            "price_from": instance.price_from,
            "price_to": instance.price_to,
            "created_at": instance.created_at.isoformat(),
            "updated_at": instance.updated_at.isoformat(),
        }

        # Create a simple non-zero vector (temporary solution)
        vector = np.random.uniform(low=0.0, high=0.1, size=1536).tolist()
        vector[0] = 1.0  # Ensure at least one non-zero value

        # Upsert to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": str(
                        instance.id
                    ),  # UUID will be automatically converted to string
                    "values": vector,
                    "metadata": metadata,
                }
            ]
        )

        print(f"Successfully saved tool {instance.title} to Pinecone")

    except Exception as e:
        logger.error(f"Error saving to Pinecone: {str(e)}")
        raise

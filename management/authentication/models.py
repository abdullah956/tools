"""Models for the authentication app."""

import uuid

from django.contrib.auth.models import AbstractUser
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


class CustomUser(AbstractUser):
    """Custom user model with additional fields."""

    AUTH_SOURCE_CHOICES = [
        ("local", "Local"),
        ("google", "Google"),
        ("linkedin", "LinkedIn"),
        ("github", "Github"),
    ]
    auth_source = models.CharField(
        max_length=50,
        choices=AUTH_SOURCE_CHOICES,
        default="local",
        null=True,
        blank=True,
    )
    trial_searches = models.IntegerField(
        default=0, help_text="Number of free trial searches available to the user"
    )
    trial_searches_total = models.IntegerField(
        default=0,
        help_text="Total number of free trial searches available to the user",
    )
    occupation = models.CharField(max_length=255)
    company = models.CharField(max_length=255, blank=True, null=True)
    is_verified = models.BooleanField(default=False)
    is_guest = models.BooleanField(default=True)
    is_early_access_user = models.BooleanField(default=False)
    login_counter = models.IntegerField(default=0)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    unique_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    access_token = models.CharField(max_length=1000, blank=True, null=True)
    refresh_token = models.CharField(max_length=1000, blank=True, null=True)
    profile_picture = models.URLField(max_length=1000, blank=True, null=True)

    USER_ROLES = (
        ("user", "Normal User"),
        ("contractor", "Contractor"),
        ("admin", "Admin"),
    )

    role = models.CharField(
        max_length=20,
        choices=USER_ROLES,
        default="user",
        help_text="User's role in the system",
    )

    # Contractor specific fields
    expertise = models.CharField(max_length=255, blank=True, null=True)
    experience = models.TextField(blank=True, null=True)
    website = models.URLField(max_length=255, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    apps_included = models.CharField(max_length=255, blank=True, null=True)
    language = models.CharField(max_length=255, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    company_name = models.CharField(max_length=255, blank=True, null=True)
    type_of_services = models.CharField(max_length=255, blank=True, null=True)
    countries_with_office_locations = models.TextField(blank=True, null=True)
    about = models.TextField(blank=True, null=True)
    availability_date = models.DateField(blank=True, null=True)
    availability_time = models.TimeField(blank=True, null=True)
    PROFILE_PIC_SOURCE_CHOICES = [
        ("google", "Google"),
        ("linkedin", "LinkedIn"),
        ("manual", "Manual Upload"),
        ("default", "Default"),
    ]
    profile_picture_source = models.CharField(
        max_length=10, choices=PROFILE_PIC_SOURCE_CHOICES, default="default"
    )

    def clean(self):
        """Validate contractor-specific fields."""
        super().clean()
        if self.role == "contractor":
            required_fields = [
                "expertise",
                "experience",
                "website",
                "language",
                "country",
                "company_name",
                "type_of_services",
                "about",
            ]
            for field in required_fields:
                if not getattr(self, field):
                    raise ValidationError(f"{field} is required for contractors")

    def save(self, *args, **kwargs):
        """Override save method to set is_verified for superusers."""
        if self.is_superuser:
            self.is_verified = True
        super().save(*args, **kwargs)

    @classmethod
    def search_contractors(cls, query):
        """Perform precise search across contractor fields."""
        if not query:
            return cls.objects.none()

        # Full-text search vector with weights
        search_vector = (
            SearchVector("first_name", weight="A")
            + SearchVector("last_name", weight="A")
            + SearchVector("email", weight="A")
            + SearchVector("username", weight="A")
            + SearchVector("company", weight="B")
            + SearchVector("occupation", weight="B")
        )
        search_query = SearchQuery(query)

        # Calculate similarity scores with coalesce to handle NULL values
        return (
            cls.objects.filter(role="contractor")
            .annotate(
                rank=SearchRank(search_vector, search_query),
                similarity_first_name=Coalesce(
                    TrigramSimilarity("first_name", query), 0.0
                ),
                similarity_last_name=Coalesce(
                    TrigramSimilarity("last_name", query), 0.0
                ),
                similarity_email=Coalesce(TrigramSimilarity("email", query), 0.0),
                similarity_username=Coalesce(TrigramSimilarity("username", query), 0.0),
                similarity_company=Coalesce(TrigramSimilarity("company", query), 0.0),
                similarity_occupation=Coalesce(
                    TrigramSimilarity("occupation", query), 0.0
                ),
                total_similarity=Coalesce(
                    (
                        TrigramSimilarity("first_name", query) * 2.0
                        + TrigramSimilarity("last_name", query) * 2.0
                        + TrigramSimilarity("email", query) * 2.0
                        + TrigramSimilarity("username", query) * 2.0
                        + TrigramSimilarity("company", query) * 1.5
                        + TrigramSimilarity("occupation", query) * 1.5
                    ),
                    0.0,
                ),
            )
            .filter(
                Q(rank__gte=0.4)  # Increased threshold for rank
                | Q(total_similarity__gte=0.4)  # Increased threshold for similarity
                | Q(
                    Q(similarity_first_name__gte=0.3)  # Name-specific threshold
                    | Q(similarity_last_name__gte=0.3)  # Name-specific threshold
                    | Q(similarity_email__gte=0.3)  # Email-specific threshold
                    | Q(similarity_username__gte=0.3)  # Username-specific threshold
                    | Q(
                        Q(similarity_company__gte=0.4)
                        & Q(similarity_occupation__gte=0.4)
                    )  # Combined field threshold
                )
            )
            .order_by("-total_similarity", "-rank", "first_name")
            .distinct()
        )

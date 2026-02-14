"""Models for Consultant Recommender app."""

import uuid

from django.conf import settings
from django.db import models


class ConsultantSearchLog(models.Model):
    """Log of consultant searches with results and metadata."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="consultant_searches",
        null=True,
        blank=True,
    )
    query = models.TextField(help_text="Search query for consultants")
    user_work_description = models.TextField(
        help_text="Description of user's work/project", blank=True
    )

    # Results metadata
    results_count = models.IntegerField(default=0)
    top_score = models.FloatField(null=True, blank=True)

    # Performance metrics
    response_time_ms = models.FloatField(null=True, blank=True)

    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ("success", "Success"),
            ("error", "Error"),
            ("no_results", "No Results"),
        ],
        default="success",
    )
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Meta options."""

        db_table = "consultant_search_logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Search: {self.query[:50]} - {self.created_at}"


class MeetingRequest(models.Model):
    """Meeting requests with contractors/consultants."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Client information
    client = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="meeting_requests",
        null=True,
        blank=True,
    )
    client_name = models.CharField(max_length=255)
    client_email = models.EmailField()
    company_name = models.CharField(max_length=255, blank=True)

    # Contractor information (can be CustomUser with role=contractor or external)
    contractor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="contractor_meetings",
        null=True,
        blank=True,
    )
    contractor_id_external = models.CharField(
        max_length=100,
        blank=True,
        help_text="External contractor ID if not a system user",
    )
    contractor_name = models.CharField(max_length=255)
    contractor_email = models.EmailField()

    # Meeting details
    preferred_date = models.DateField()
    preferred_time = models.TimeField()
    project_description = models.TextField()

    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("confirmed", "Confirmed"),
            ("cancelled", "Cancelled"),
            ("completed", "Completed"),
        ],
        default="pending",
    )

    # Notes
    client_notes = models.TextField(blank=True)
    contractor_notes = models.TextField(blank=True)
    admin_notes = models.TextField(blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    confirmed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        """Meta options."""

        db_table = "meeting_requests"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["client", "status"]),
            models.Index(fields=["contractor", "status"]),
            models.Index(fields=["status", "preferred_date"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Meeting: {self.client_name} -> {self.contractor_name} on {self.preferred_date}"


class ConsultantRecommendation(models.Model):
    """Track consultant recommendations made to users."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    search_log = models.ForeignKey(
        ConsultantSearchLog,
        on_delete=models.CASCADE,
        related_name="recommendations",
    )

    # Consultant can be a CustomUser with role=contractor or external consultant
    contractor_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="recommendations_as_contractor",
        null=True,
        blank=True,
    )
    consultant_external_id = models.CharField(
        max_length=100,
        blank=True,
        help_text="External consultant ID from Pinecone",
    )

    # Recommendation details
    relevance_score = models.FloatField()
    rank = models.IntegerField(help_text="Position in recommendation list")

    # Interaction tracking
    viewed = models.BooleanField(default=False)
    clicked = models.BooleanField(default=False)
    contacted = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    viewed_at = models.DateTimeField(null=True, blank=True)
    clicked_at = models.DateTimeField(null=True, blank=True)
    contacted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        """Meta options."""

        db_table = "consultant_recommendations"
        ordering = ["search_log", "rank"]
        indexes = [
            models.Index(fields=["search_log", "rank"]),
            models.Index(fields=["contractor_user", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        contractor_name = (
            self.contractor_user.get_full_name()
            if self.contractor_user
            else self.consultant_external_id
        )
        return f"Recommendation: {contractor_name} (rank {self.rank})"


class WorkflowConsultants(models.Model):
    """Cache consultants for specific workflows to avoid repeated searches."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow_id = models.CharField(
        max_length=36,
        help_text="ID of the workflow these consultants are cached for",
        db_index=True,
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workflow_consultants",
        help_text="User who owns this workflow consultant cache",
    )

    # Search parameters used to generate these consultants
    query = models.TextField(help_text="Search query used to find consultants")
    user_work_description = models.TextField(
        help_text="User work description used in search", blank=True
    )

    # Cached consultant data as JSON list
    consultants_data = models.JSONField(
        help_text="JSON array of consultant objects with all their details"
    )

    # Metadata
    consultants_count = models.IntegerField(default=0)
    search_metadata = models.JSONField(
        default=dict,
        help_text="Additional metadata from the search (total_found, response_time, etc.)",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options."""

        db_table = "workflow_consultants"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["workflow_id", "user"]),
            models.Index(fields=["user", "updated_at"]),
        ]
        # Ensure one consultant cache per workflow per user
        unique_together = ["workflow_id", "user"]

    def __str__(self):
        """String representation."""
        return f"Consultants for workflow {self.workflow_id} - {self.consultants_count} consultants"

    def save(self, *args, **kwargs):
        """Update consultants_count when saving."""
        if self.consultants_data:
            self.consultants_count = len(self.consultants_data)
        super().save(*args, **kwargs)

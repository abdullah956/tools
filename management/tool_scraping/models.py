"""Models for tool scraping."""

import uuid

from django.db import models
from django.utils.translation import gettext_lazy as _


class ScrapingJob(models.Model):
    """Tracks the overall CSV upload and processing job."""

    class Status(models.TextChoices):
        PENDING = "PENDING", _("Pending")
        PROCESSING = "PROCESSING", _("Processing")
        COMPLETED = "COMPLETED", _("Completed")
        FAILED = "FAILED", _("Failed")

    class JobType(models.TextChoices):
        CSV_UPLOAD = "CSV_UPLOAD", _("CSV Upload")
        INTERNET_DISCOVERY = "INTERNET_DISCOVERY", _("Internet Discovery")

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job_type = models.CharField(
        max_length=50, choices=JobType.choices, default=JobType.CSV_UPLOAD
    )
    payload = models.JSONField(default=dict, blank=True, help_text="Arbitrary job data")
    file = models.FileField(upload_to="csv_uploads/", null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    logs = models.JSONField(default=list, blank=True)

    class Meta:
        """Meta class for ScrapingJob."""

        app_label = "tool_scraping"

    def __str__(self):
        """Return string representation of ScrapingJob."""
        return f"Job {self.id} - {self.status}"


class ToolSite(models.Model):
    """Represents a single tool/website from the CSV."""

    class Status(models.TextChoices):
        PENDING = "PENDING", _("Pending")
        INVALID = "INVALID", _("Invalid URL")
        QUEUED = "QUEUED", _("Queued for Scraping")
        SCRAPED = "SCRAPED", _("Scraped")
        PROCESSED = "PROCESSED", _("Processed")
        INDEXED = "INDEXED", _("Indexed")
        FAILED = "FAILED", _("Failed")

    job = models.ForeignKey(ScrapingJob, on_delete=models.CASCADE, related_name="sites")
    csv_row_nr = models.IntegerField(help_text="Row number in the original CSV")

    # Core fields from CSV - dedicated columns for common fields
    website = models.URLField(max_length=2000, help_text="Primary website URL")
    title = models.CharField(max_length=500, blank=True)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=255, blank=True)
    master_category = models.CharField(max_length=255, blank=True)

    # Status tracking
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    # Content
    raw_html = models.TextField(blank=True, null=True)
    clean_html = models.TextField(blank=True, null=True)

    # All extracted fields (JSONB) - stores ~200 metadata fields from LLM
    fields = models.JSONField(
        default=dict, blank=True, help_text="All extracted metadata fields"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta class for ToolSite."""

        app_label = "tool_scraping"

    def __str__(self):
        """Return string representation of ToolSite."""
        return f"{self.title} ({self.website})"


class SitePage(models.Model):
    """Individual pages discovered via sitemap for a ToolSite."""

    class Status(models.TextChoices):
        PENDING = "PENDING", _("Pending")
        SCRAPED = "SCRAPED", _("Scraped")
        FAILED = "FAILED", _("Failed")

    site = models.ForeignKey(ToolSite, on_delete=models.CASCADE, related_name="pages")
    url = models.URLField(max_length=2000)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )

    raw_html = models.TextField(blank=True, null=True)
    clean_html = models.TextField(blank=True, null=True)
    page_text = models.TextField(blank=True, null=True)

    discovered_at = models.DateTimeField(auto_now_add=True)
    scraped_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        """Meta class for SitePage."""

        app_label = "tool_scraping"
        unique_together = ["site", "url"]

    def __str__(self):
        """Return string representation of SitePage."""
        return self.url


class CombinedText(models.Model):
    """Stores combined text from all pages of a site with UUID for Pinecone."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.OneToOneField(
        ToolSite, on_delete=models.CASCADE, related_name="combined_text_record"
    )
    combined_text = models.TextField(help_text="Combined text from all site pages")
    char_count = models.IntegerField(help_text="Total character count")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta class for CombinedText."""

        app_label = "tool_scraping"

    def __str__(self):
        """Return string representation of CombinedText."""
        return f"{self.site.title} - Combined Text ({self.char_count} chars)"

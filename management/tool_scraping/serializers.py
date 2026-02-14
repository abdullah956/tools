"""Serializers for tool scraping models."""

from rest_framework import serializers

from .models import CombinedText, ScrapingJob, SitePage, ToolSite


class InternetDiscoveredToolSerializer(serializers.Serializer):
    """Serializer for tools discovered via internet search."""

    title = serializers.CharField(required=True, max_length=500)
    description = serializers.CharField(required=True, allow_blank=True)
    website = serializers.URLField(required=True, max_length=2000)
    category = serializers.CharField(required=False, allow_blank=True, default="")
    features = serializers.CharField(required=False, allow_blank=True, default="")
    source = serializers.CharField(
        required=False, allow_blank=True, default="Internet Search"
    )
    relevance_score = serializers.IntegerField(
        required=False, default=0, min_value=0, max_value=10
    )

    # Optional social media fields
    twitter = serializers.URLField(required=False, allow_blank=True, default="")
    facebook = serializers.URLField(required=False, allow_blank=True, default="")
    linkedin = serializers.URLField(required=False, allow_blank=True, default="")
    instagram = serializers.URLField(required=False, allow_blank=True, default="")


class BulkInternetToolsSerializer(serializers.Serializer):
    """Serializer for bulk submission of internet-discovered tools."""

    tools = InternetDiscoveredToolSerializer(many=True, required=True)
    source_query = serializers.CharField(required=False, allow_blank=True, default="")


class ScrapingJobSerializer(serializers.ModelSerializer):
    """Serializer for ScrapingJob model."""

    # Make these fields optional for the upload endpoint
    payload = serializers.JSONField(required=False, allow_null=True, default=dict)
    job_type = serializers.CharField(required=False, default="CSV_UPLOAD")

    class Meta:
        model = ScrapingJob
        fields = [
            "id",
            "job_type",
            "payload",
            "file",
            "status",
            "started_at",
            "finished_at",
            "created_at",
            "updated_at",
            "logs",
        ]
        read_only_fields = [
            "id",
            "status",
            "started_at",
            "finished_at",
            "created_at",
            "updated_at",
            "logs",
        ]


class ToolSiteSerializer(serializers.ModelSerializer):
    """Serializer for ToolSite model."""

    class Meta:
        model = ToolSite
        fields = "__all__"


class SitePageSerializer(serializers.ModelSerializer):
    """Serializer for SitePage model."""

    class Meta:
        model = SitePage
        fields = "__all__"


class CombinedTextSerializer(serializers.ModelSerializer):
    """Serializer for CombinedText model."""

    class Meta:
        model = CombinedText
        fields = [
            "id",
            "site",
            "combined_text",
            "char_count",
            "created_at",
            "updated_at",
        ]

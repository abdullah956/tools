"""Serializers for the submit resource form app."""

from rest_framework import serializers

from .models import SubmitResourceForm


class SubmitResourceFormSerializer(serializers.ModelSerializer):
    """Serializer for the SubmitResourceForm model."""

    class Meta:
        """Meta class for SubmitResourceFormSerializer."""

        model = SubmitResourceForm
        fields = ["id", "url", "title", "description", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

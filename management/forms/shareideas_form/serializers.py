"""Serializers for the share ideas form app."""

from rest_framework import serializers

from .models import ShareIdeasForm


class ShareIdeasFormSerializer(serializers.ModelSerializer):
    """Serializer for the ShareIdeasForm model."""

    class Meta:
        """Meta class for ShareIdeasFormSerializer."""

        model = ShareIdeasForm
        fields = ["id", "name", "feedback", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

"""CHAT SERIALIZERS."""
from rest_framework import serializers

from .models import ChatMessage, ChatSession


class ChatMessageSerializer(serializers.ModelSerializer):
    """CHAT MESSAGE SERIALIZER."""

    class Meta:
        """Meta class."""

        model = ChatMessage
        fields = ["id", "role", "content", "timestamp"]
        read_only_fields = ["id", "timestamp"]


class ChatSessionSerializer(serializers.ModelSerializer):
    """CHAT SESSION SERIALIZER."""

    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        """Meta class."""

        model = ChatSession
        fields = [
            "id",
            "workflow_data",
            "title",
            "created_at",
            "updated_at",
            "messages",
            "workflow_id",
            "chat_type",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class ChatSessionListSerializer(serializers.ModelSerializer):
    """CHAT SESSION LIST SERIALIZER."""

    class Meta:
        """Meta class."""

        model = ChatSession
        fields = ["chat_type", "workflow_id"]


class ChatInputSerializer(serializers.Serializer):
    """CHAT INPUT SERIALIZER."""

    message = serializers.CharField(required=True)
    workflow = serializers.JSONField(required=True)
    session_id = serializers.UUIDField(required=False, allow_null=True)
    workflow_id = serializers.UUIDField(required=False)
    chat_type = serializers.CharField(required=False, default="general")

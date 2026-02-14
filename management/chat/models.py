"""Models for chat functionality."""
import uuid

from django.conf import settings
from django.db import models

CHAT_TYPE_CHOICES = [
    ("workflow", "workflow"),
    ("general", "general"),
]


class ChatSession(models.Model):
    """Model to track chat sessions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chat_sessions"
    )
    workflow_data = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.CharField(max_length=255, blank=True)
    workflow_id = models.UUIDField(editable=False, null=True, blank=True)
    chat_type = models.CharField(
        max_length=255, choices=CHAT_TYPE_CHOICES, default="workflow"
    )

    class Meta:
        """Meta class."""

        ordering = ["-updated_at"]


class ChatMessage(models.Model):
    """Model to store chat messages."""

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Meta class."""

        ordering = ["timestamp"]

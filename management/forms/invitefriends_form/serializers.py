"""Serializers for the invite friends form app."""

from rest_framework import serializers

from .models import InviteFriendsForm


class InviteFriendsFormSerializer(serializers.ModelSerializer):
    """Serializer for the InviteFriendsForm model."""

    class Meta:
        """Meta class for InviteFriendsFormSerializer."""

        model = InviteFriendsForm
        fields = ["id", "emails", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

    def validate_emails(self, value):
        """Validate the emails array."""
        if not value:
            raise serializers.ValidationError("At least one email is required.")
        if len(value) > 3:
            raise serializers.ValidationError("Maximum 3 email addresses allowed.")
        return value

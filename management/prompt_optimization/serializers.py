"""Serializers for the prompt optimization app."""

from rest_framework import serializers

from .models import UserQuery


class UserQuerySerializer(serializers.ModelSerializer):
    """Serializer for the UserQuery model."""

    user_unique_id = serializers.SerializerMethodField()

    def get_user_unique_id(self, obj):
        """Get the unique_id of the user."""
        return str(obj.user.unique_id)

    class Meta:
        """Meta class for UserQuerySerializer."""

        model = UserQuery
        fields = [
            "id",
            "original_query",
            "refined_query",
            "created_at",
            "workflow",
            "user_unique_id",
        ]
        read_only_fields = ["refined_query", "created_at", "user", "user_unique_id"]

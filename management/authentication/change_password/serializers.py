"""Password change serializers."""

from rest_framework import serializers


class PasswordChangeSerializer(serializers.Serializer):
    """Password change serializer."""

    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(required=True)

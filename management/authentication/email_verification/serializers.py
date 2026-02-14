"""Email verification serializers."""

from rest_framework import serializers


class EmailVerificationSerializer(serializers.Serializer):
    """Email verification serializer."""

    token = serializers.CharField(max_length=255, required=True)
    email = serializers.EmailField(required=True)

"""Set password serializers."""

from rest_framework import serializers


class SetPasswordSerializer(serializers.Serializer):
    """Set password serializer."""

    new_password = serializers.CharField(required=True)

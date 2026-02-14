"""Serializers for the guest user app."""

from rest_framework import serializers

from .models import GuestUser


class GuestUserSerializer(serializers.ModelSerializer):
    """Serializer for the GuestUser model."""

    class Meta:
        """Meta class for GuestUserSerializer."""

        model = GuestUser
        fields = ["id", "ip_address", "no_of_requests", "refresh_token", "access_token"]
        read_only_fields = [
            "id",
            "ip_address",
            "no_of_requests",
            "refresh_token",
            "access_token",
        ]

"""Serializer for password reset requests."""

from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from authentication.models import CustomUser


class PasswordResetRequestSerializer(serializers.Serializer):
    """Serializer for password reset requests."""

    email = serializers.EmailField()

    def validate_email(self, value):
        """Validate the email field."""
        if not CustomUser.objects.filter(email=value, is_active=True).exists():
            raise serializers.ValidationError("User with this email does not exist.")
        return value


class PasswordResetConfirmSerializer(serializers.Serializer):
    """Serializer for password reset confirmation."""

    token = serializers.CharField(required=True)
    uid = serializers.CharField(required=True)
    new_password1 = serializers.CharField(required=True)
    new_password2 = serializers.CharField(required=True)

    def validate_new_password1(self, value):
        """Validate the new password field."""
        validate_password(value)
        return value

    def validate(self, data):
        """Validate the new password fields."""
        if data["new_password1"] != data["new_password2"]:
            raise serializers.ValidationError("The two password fields didn't match.")
        return data

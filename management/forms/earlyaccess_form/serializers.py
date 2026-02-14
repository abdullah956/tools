"""Serializers for the early access form app."""

from rest_framework import serializers

from .models import EarlyAccessForm


class EarlyAccessFormSerializer(serializers.ModelSerializer):
    """Serializer for the EarlyAccessForm model."""

    class Meta:
        """Meta class for EarlyAccessFormSerializer."""

        model = EarlyAccessForm
        fields = [
            "id",
            "email",
            "created_at",
            "updated_at",
            "product_id",
            "has_paid",
            "payment_status",
            "payment_date",
            "email_verification_token",
            "token_created_at",
            "is_email_verified",
        ]
        read_only_fields = [
            "created_at",
            "updated_at",
            "id",
            "has_paid",
            "payment_status",
            "payment_date",
            "email_verification_token",
            "token_created_at",
            "is_email_verified",
        ]
        extra_kwargs = {"product_id": {"write_only": True}}

    def validate_email(self, value):
        """Validate that the email hasn't already been used with a paid subscription."""
        try:
            existing_form = EarlyAccessForm.objects.get(email=value)
            if existing_form.has_paid:
                raise serializers.ValidationError(
                    "This email has already been used for early access."
                )
        except EarlyAccessForm.DoesNotExist:
            pass
        return value

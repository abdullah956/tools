"""Serializers for the newsletter subscriber form app."""

from rest_framework import serializers

from .models import NewsletterSubscriberForm


class NewsletterSubscriberFormSerializer(serializers.ModelSerializer):
    """Serializer for the NewsletterSubscriberForm model."""

    class Meta:
        """Meta class for NewsletterSubscriberFormSerializer."""

        model = NewsletterSubscriberForm
        fields = ["id", "email", "created_at", "updated_at"]
        read_only_fields = ["created_at", "updated_at"]

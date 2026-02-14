"""This module contains the serializers for the tools."""

from rest_framework import serializers

from .models import Tool


class ToolSerializer(serializers.ModelSerializer):
    """This class contains the serializer for the Tool model."""

    class Meta:
        """This class contains the Meta class for the ToolSerializer."""

        model = Tool
        fields = [
            "id",
            "title",
            "description",
            "category",
            "features",
            "tags",
            "website",
            "twitter",
            "facebook",
            "linkedin",
            "tiktok",
            "youtube",
            "instagram",
            "price_from",
            "price_to",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def create(self, validated_data):
        """This method creates a new tool."""
        tool = Tool.objects.create(**validated_data)
        return tool

    def update(self, instance, validated_data):
        """This method updates a tool."""
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

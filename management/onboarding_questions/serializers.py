"""Serializers for the onboarding questions app."""

from rest_framework import serializers

from .models import OnboardingQuestion


class OnboardingQuestionSerializer(serializers.ModelSerializer):
    """Serializer for the OnboardingQuestion model."""

    user_unique_id = serializers.SerializerMethodField()

    def get_user_unique_id(self, obj):
        """Get the unique_id of the user."""
        return str(obj.user.unique_id)

    class Meta:
        """Meta class for OnboardingQuestionSerializer."""

        model = OnboardingQuestion
        fields = [
            "id",
            "user_unique_id",
            "company_info",
            "website_or_linkedin",
            "experience_level",
            "goals",
        ]
        read_only_fields = ["user_unique_id"]

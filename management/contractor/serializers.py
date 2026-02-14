"""Serializers for the contractor app."""

from rest_framework import serializers


class ConsultantSearchSerializer(serializers.Serializer):
    """Serializer for searching for consultants."""

    query = serializers.CharField(
        required=True, help_text="Search query for finding relevant consultants"
    )
    user_work_description = serializers.CharField(
        required=True,
        help_text="Description of the user's work or project requirements",
    )


class MeetingBookingSerializer(serializers.Serializer):
    """Serializer for booking meetings with contractors."""

    contractor_id = serializers.CharField()
    preferred_date = serializers.DateField()
    preferred_time = serializers.TimeField()
    project_description = serializers.CharField()
    client_name = serializers.CharField()
    client_email = serializers.EmailField()
    company_name = serializers.CharField(required=False, default="")

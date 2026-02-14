"""Serializers for Consultant Recommender app."""

from rest_framework import serializers

from authentication.models import CustomUser

from .models import (
    ConsultantRecommendation,
    ConsultantSearchLog,
    MeetingRequest,
    WorkflowConsultants,
)


class ConsultantSearchSerializer(serializers.Serializer):
    """Serializer for consultant search queries."""

    workflow_id = serializers.CharField(
        required=True,
        help_text="Workflow ID to fetch refined query and search consultants",
    )


class ConsultantSearchLogSerializer(serializers.ModelSerializer):
    """Serializer for Consultant Search Log."""

    user_email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        """Meta options."""

        model = ConsultantSearchLog
        fields = [
            "id",
            "user",
            "user_email",
            "query",
            "user_work_description",
            "results_count",
            "top_score",
            "response_time_ms",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "user_email",
            "results_count",
            "top_score",
            "response_time_ms",
            "status",
            "error_message",
            "created_at",
        ]


class MeetingRequestSerializer(serializers.ModelSerializer):
    """Serializer for Meeting Request."""

    client_email_field = serializers.EmailField(source="client.email", read_only=True)
    contractor_email_field = serializers.EmailField(
        source="contractor.email", read_only=True, allow_null=True
    )

    class Meta:
        """Meta options."""

        model = MeetingRequest
        fields = [
            "id",
            "client",
            "client_email_field",
            "client_name",
            "client_email",
            "company_name",
            "contractor",
            "contractor_email_field",
            "contractor_id_external",
            "contractor_name",
            "contractor_email",
            "preferred_date",
            "preferred_time",
            "project_description",
            "status",
            "client_notes",
            "contractor_notes",
            "admin_notes",
            "created_at",
            "updated_at",
            "confirmed_at",
        ]
        read_only_fields = [
            "id",
            "client",
            "client_email_field",
            "contractor_email_field",
            "created_at",
            "updated_at",
            "confirmed_at",
        ]


class MeetingBookingSerializer(serializers.Serializer):
    """Serializer for booking meetings with contractors."""

    contractor_id = serializers.CharField(
        help_text="ID of the contractor (user ID or external ID)",
    )
    preferred_date = serializers.DateField(
        help_text="Preferred meeting date",
    )
    preferred_time = serializers.TimeField(
        help_text="Preferred meeting time",
    )
    project_description = serializers.CharField(
        help_text="Description of the project",
    )
    client_name = serializers.CharField(
        help_text="Client name",
    )
    client_email = serializers.EmailField(
        help_text="Client email",
    )
    company_name = serializers.CharField(
        required=False,
        default="",
        help_text="Company name (optional)",
    )


class ConsultantRecommendationSerializer(serializers.ModelSerializer):
    """Serializer for Consultant Recommendation."""

    consultant_name = serializers.SerializerMethodField()
    consultant_email = serializers.SerializerMethodField()

    class Meta:
        """Meta options."""

        model = ConsultantRecommendation
        fields = [
            "id",
            "search_log",
            "contractor_user",
            "consultant_external_id",
            "consultant_name",
            "consultant_email",
            "relevance_score",
            "rank",
            "viewed",
            "clicked",
            "contacted",
            "created_at",
            "viewed_at",
            "clicked_at",
            "contacted_at",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "viewed_at",
            "clicked_at",
            "contacted_at",
        ]

    def get_consultant_name(self, obj):
        """Get consultant name."""
        if obj.contractor_user:
            return obj.contractor_user.get_full_name() or obj.contractor_user.username
        return obj.consultant_external_id

    def get_consultant_email(self, obj):
        """Get consultant email."""
        if obj.contractor_user:
            return obj.contractor_user.email
        return None


class ContractorSerializer(serializers.ModelSerializer):
    """Serializer for contractors (CustomUser with role=contractor)."""

    class Meta:
        """Meta options."""

        model = CustomUser
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "role",
            "expertise",
            "experience",
            "website",
            "phone",
            "apps_included",
            "language",
            "country",
            "company_name",
            "type_of_services",
            "countries_with_office_locations",
            "about",
            "availability_date",
            "availability_time",
            "profile_picture",
        ]
        read_only_fields = ["id", "role"]


class WorkflowConsultantsSerializer(serializers.ModelSerializer):
    """Serializer for WorkflowConsultants model."""

    class Meta:
        """Meta options."""

        model = WorkflowConsultants
        fields = [
            "id",
            "workflow_id",
            "user",
            "query",
            "user_work_description",
            "consultants_data",
            "consultants_count",
            "search_metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "consultants_count",
            "created_at",
            "updated_at",
        ]


class ConsultantDetailSerializer(serializers.Serializer):
    """Serializer for individual consultant data structure."""

    name = serializers.CharField(required=False, allow_null=True)
    expertise = serializers.CharField(required=False, allow_null=True)
    experience = serializers.CharField(required=False, allow_null=True)
    website = serializers.CharField(required=False, allow_null=True)
    phone = serializers.CharField(required=False, allow_null=True)
    gmail = serializers.EmailField(required=False, allow_null=True)
    apps_included = serializers.CharField(required=False, allow_null=True)
    language = serializers.CharField(required=False, allow_null=True)
    country = serializers.CharField(required=False, allow_null=True)
    company_name = serializers.CharField(required=False, allow_null=True)
    type_of_services = serializers.CharField(required=False, allow_null=True)
    countries_with_office_locations = serializers.CharField(
        required=False, allow_null=True
    )
    about = serializers.CharField(required=False, allow_null=True)
    date = serializers.CharField(required=False, allow_null=True)
    time = serializers.CharField(required=False, allow_null=True)

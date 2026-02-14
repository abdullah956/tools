"""Serializers for the authentication app."""

import logging

import numpy as np
from envs.env_loader import EnvLoader
from pinecone import Pinecone, ServerlessSpec  # Updated import
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import CustomUser

logger = logging.getLogger(__name__)
env_loader = EnvLoader()


class CustomUserSerializer(serializers.ModelSerializer):
    """Serializer for the CustomUser model."""

    password = serializers.CharField(write_only=True, required=False)
    occupation = serializers.CharField(required=False)
    role = serializers.ChoiceField(choices=CustomUser.USER_ROLES, default="user")
    trial_searches = serializers.IntegerField(read_only=True)

    class Meta:
        """Meta class for CustomUserSerializer."""

        model = CustomUser
        read_only_fields = [
            "id",
            "is_guest",
            "ip_address",
            "unique_id",
            "login_counter",
            "trial_searches",
        ]
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "occupation",
            "company",
            "password",
            "auth_source",
            "ip_address",
            "unique_id",
            "login_counter",
            "role",
            "trial_searches",
            "trial_searches_total",
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
        extra_kwargs = {
            "password": {"write_only": True},
        }

    def validate(self, data):
        """Validate contractor-specific fields."""
        if data.get("role") == "contractor":
            required_fields = [
                "expertise",
                "experience",
                "website",
                "language",
                "country",
                "company_name",
                "type_of_services",
                "about",
            ]
            for field in required_fields:
                if not data.get(field):
                    raise serializers.ValidationError(
                        f"{field} is required for contractors"
                    )
        return data

    def create(self, validated_data):
        """Create a new CustomUser instance."""
        # Extract contractor-specific fields
        contractor_fields = {
            "expertise": validated_data.pop("expertise", None),
            "experience": validated_data.pop("experience", None),
            "website": validated_data.pop("website", None),
            "phone": validated_data.pop("phone", None),
            "apps_included": validated_data.pop("apps_included", None),
            "language": validated_data.pop("language", None),
            "country": validated_data.pop("country", None),
            "company_name": validated_data.pop("company_name", None),
            "type_of_services": validated_data.pop("type_of_services", None),
            "countries_with_office_locations": validated_data.pop(
                "countries_with_office_locations", None
            ),
            "about": validated_data.pop("about", None),
            "availability_date": validated_data.pop("availability_date", None),
            "availability_time": validated_data.pop("availability_time", None),
        }

        # Create user with basic fields
        user = CustomUser.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
            first_name=validated_data.get("first_name", ""),
            last_name=validated_data.get("last_name", ""),
            auth_source=validated_data.get("auth_source", "local"),
            occupation=validated_data.get("occupation", ""),
            company=validated_data.get("company", ""),
            role=validated_data.get("role", "user"),
        )

        # If user is a contractor, save contractor-specific fields
        if user.role == "contractor":
            for field, value in contractor_fields.items():
                if value is not None:
                    setattr(user, field, value)
            user.save()

        # Save user data to Pinecone
        try:
            self.save_to_pinecone(user)
        except Exception as e:
            logger.error(f"Failed to save user to Pinecone: {e}")

        return user

    def save_to_pinecone(self, user):
        """Save user data to Pinecone."""
        try:
            # Initialize Pinecone client
            pinecone_client = Pinecone(
                api_key=env_loader.pinecone_api_key,
            )

            # Get or create the index
            index_name = env_loader.pinecone_contractor_index
            indexes = pinecone_client.list_indexes()

            if index_name not in [idx.name for idx in indexes]:
                # Create index with proper spec
                pinecone_client.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            # Connect to the index
            index = pinecone_client.Index(index_name)

            # Enhanced metadata for contractors - handle null values with defaults
            metadata = {
                "id": str(user.id),
                "email": user.email or "",
                "role": user.role or "user",
                "occupation": user.occupation or "",
                "company": user.company or "",
                "auth_source": user.auth_source or "local",
                "is_contractor": user.role == "contractor",
                "trial_searches": user.trial_searches,
                "trial_searches_total": user.trial_searches_total,
            }

            # Add contractor-specific fields to metadata if user is a contractor
            if user.role == "contractor":
                contractor_fields = {
                    "expertise": user.expertise or "",
                    "experience": user.experience or "",
                    "website": user.website or "",
                    "phone": user.phone or "",
                    "apps_included": user.apps_included or "",
                    "language": user.language or "",
                    "country": user.country or "",
                    "company_name": user.company_name or "",
                    "type_of_services": user.type_of_services or "",
                    "countries_with_office_locations": (
                        user.countries_with_office_locations or ""
                    ),
                    "about": user.about or "",
                    # Convert dates and times to strings with fallbacks
                    "availability_date": (
                        user.availability_date.strftime("%Y-%m-%d")
                        if user.availability_date
                        else ""
                    ),
                    "availability_time": (
                        user.availability_time.strftime("%H:%M")
                        if user.availability_time
                        else ""
                    ),
                }
                metadata.update(contractor_fields)

            # Create a simple non-zero vector
            vector = np.random.uniform(low=0.0, high=0.1, size=1536).tolist()
            vector[0] = 1.0  # Ensure at least one non-zero value

            # Upsert the user data with str(user.id)
            index.upsert(
                vectors=[
                    {
                        "id": str(user.id),
                        "values": vector,
                        "metadata": metadata,
                    }
                ]
            )

            print(f"Successfully saved user {user.email} to Pinecone")

        except Exception as e:
            logger.error(f"Error saving to Pinecone: {str(e)}")
            raise

    def update(self, instance, validated_data):
        """Update an existing CustomUser instance."""
        # Handle password update separately
        password = validated_data.pop("password", None)
        if password:
            instance.set_password(password)

        # Update other fields
        return super().update(instance, validated_data)


class CodeSerializer(serializers.Serializer):
    """Serializer for handling OAuth authorization codes from providers like Google and LinkedIn."""

    code = serializers.CharField(
        required=True,
        help_text="Authorization code from OAuth provider (Google, LinkedIn, etc.)",
    )


class LoginSerializer(serializers.Serializer):
    """Serializer for user login."""

    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True, write_only=True)

    def validate(self, attrs):
        """Validate and authenticate the user."""
        email = attrs.get("email")
        password = attrs.get("password")

        if email and password:
            try:
                # Get user by email
                user = CustomUser.objects.get(email=email)

                # Check password
                if not user.check_password(password):
                    raise serializers.ValidationError(
                        {"error": "Invalid email or password."}
                    )

                # Check if user is verified (skip in development)
                if not user.is_verified and env_loader.env_type != "development":
                    raise serializers.ValidationError(
                        {"error": "Please verify your email to log in."}
                    )

                # Generate tokens
                refresh = RefreshToken.for_user(user)
                refresh["unique_id"] = str(user.unique_id)

                # Update login counter
                user.login_counter += 1
                user.save()

                return {
                    "user": user,
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                }

            except CustomUser.DoesNotExist:
                raise serializers.ValidationError(
                    {"error": "Invalid email or password."}
                )

        raise serializers.ValidationError({"error": "Email and password are required."})


class PDFUploadSerializer(serializers.Serializer):
    """Serializer for handling PDF file uploads."""

    pdf_file = serializers.FileField(
        required=True, help_text="PDF file containing contractor resume/CV"
    )

    def validate_pdf_file(self, value):
        """Validate that the uploaded file is a PDF."""
        # Check if file is provided
        if not value:
            raise serializers.ValidationError("No file was submitted.")

        # Check file extension
        if not value.name.endswith(".pdf"):
            raise serializers.ValidationError("Only PDF files are allowed.")

        # Check file size (limit to 10MB)
        if value.size > 10 * 1024 * 1024:  # 10MB in bytes
            raise serializers.ValidationError("File size cannot exceed 10MB.")

        return value


class ProfilePictureSerializer(serializers.Serializer):
    """Serializer for profile picture upload."""

    profile_picture = serializers.ImageField(
        required=True, help_text="Profile picture file (JPEG, JPG, PNG only, max 5MB)"
    )

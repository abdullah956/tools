"""Custom views for token handling in the Victor project."""

from contextlib import suppress

from django.conf import settings
from jwt import decode
from rest_framework import serializers
from rest_framework.response import Response
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView, TokenVerifyView

from authentication.models import CustomUser


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Custom serializer for obtaining JWT tokens."""

    def validate(self, attrs):
        """Validate user credentials and obtain JWT tokens."""
        user = None
        if "username" in attrs:
            with suppress(CustomUser.DoesNotExist):
                user = CustomUser.objects.get(username=attrs["username"])
        if not user and "email" in attrs:
            with suppress(CustomUser.DoesNotExist):
                user = CustomUser.objects.get(email=attrs["email"])
        if not user and "unique_id" in attrs:
            with suppress(CustomUser.DoesNotExist):
                user = CustomUser.objects.get(unique_id=attrs["unique_id"])

        if not user:
            raise serializers.ValidationError("User does not exist")

        data = super().validate(attrs)

        # Add unique_id to token payload
        refresh = self.get_token(self.user)
        refresh["unique_id"] = str(
            self.user.unique_id
        )  # Convert to string to ensure serialization

        # Update tokens with the modified payload
        data["refresh"] = str(refresh)
        data["access"] = str(refresh.access_token)

        return data

    @classmethod
    def get_token(cls, user):
        """Get token with custom claims."""
        token = super().get_token(user)
        # Add unique_id to token payload
        token["unique_id"] = str(
            user.unique_id
        )  # Convert to string to ensure serialization
        return token


class CustomTokenObtainPairView(TokenObtainPairView):
    """Custom view for obtaining JWT tokens."""

    serializer_class = CustomTokenObtainPairSerializer

    def post(self, request, *args, **kwargs):
        """Override post method to ensure unique_id is included in token."""
        response = super().post(request, *args, **kwargs)

        # For debugging purposes
        if response.status_code == 200 and hasattr(self, "get_serializer"):
            serializer = self.get_serializer(data=request.data)
            if serializer.is_valid():
                user = serializer.user
                print(f"Generated token {user.username}, unique_id: {user.unique_id}")

        return response


class CustomTokenVerifyView(TokenVerifyView):
    """Custom view for verifying JWT tokens."""

    def post(self, request, *args, **kwargs):
        """Verify the provided JWT token."""
        serializer = self.get_serializer(data=request.data)

        try:
            serializer.is_valid(raise_exception=True)
        except TokenError as e:
            return Response({"error": "Invalid token", "details": str(e)}, status=400)

        token = request.data.get("token")
        if not token:
            return Response({"error": "Token is required"}, status=400)

        try:
            decoded_token = decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            print("decoded_token", decoded_token)
            user_id = decoded_token.get("user_id")
        except Exception as e:
            return Response(
                {"error": "Token decoding failed", "details": str(e)}, status=400
            )

        try:
            user = CustomUser.objects.get(id=user_id)
            user_data = {
                "username": user.username,
                "email": user.email,
                "unique_id": user.unique_id,
            }
        except CustomUser.DoesNotExist:
            return Response({"error": "Invalid token"}, status=400)

        return Response({"token_valid": True, "user": user_data})


def custom_user_authentication_rule(user):
    """Custom authentication rule for JWT tokens."""
    return user is not None and user.is_active

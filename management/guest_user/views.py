"""Views for managing guest user interactions."""

import requests
from drf_spectacular.utils import extend_schema
from envs.env_loader import env_loader
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken

from .models import GuestUser
from .serializers import GuestUserSerializer


class QuerySerializer(serializers.Serializer):
    """Serializer for handling queries sent to the AI tool recommender."""

    query = serializers.CharField(
        required=True, help_text="The query to send to the AI tool recommender."
    )
    access_token = serializers.CharField(
        required=True, help_text="The access token to send to the AI tool recommender."
    )


class GuestUserViewSet(viewsets.ModelViewSet):
    """ViewSet for managing guest users."""

    queryset = GuestUser.objects.all()
    serializer_class = GuestUserSerializer
    permission_classes = [AllowAny]

    def get_client_ip(self, request):
        """Get client IP address from request."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0]
        else:
            ip = request.META.get("REMOTE_ADDR")
        return ip or None  # Return None instead of a hardcoded IP

    def send_query_to_ai(self, query):
        """Send a query to the AI tool recommender and return the response."""
        try:
            response = requests.post(
                env_loader.guest_user_query_url,
                json={"query": query},
                timeout=10,  # Add a timeout of 10 seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def create(self, request, *args, **kwargs):
        """Create or retrieve a guest user and generate JWT tokens."""
        # Get client IP
        ip_address = self.get_client_ip(request)

        # Check if the guest user already exists
        guest_user, created = GuestUser.objects.get_or_create(ip_address=ip_address)

        # Generate JWT tokens
        refresh = RefreshToken.for_user(guest_user)
        token_data = {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }

        # Save tokens in the database
        guest_user.refresh_token = token_data["refresh"]
        guest_user.access_token = token_data["access"]
        guest_user.save()

        return Response(
            {
                "message": "Guest user created successfully",
                "ip_address": ip_address,
                **token_data,
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )

    @extend_schema(
        request=QuerySerializer,
        responses={200: "AI tool response"},
    )
    @action(detail=False, methods=["post"], url_path="send-query")
    def send_query(self, request):
        """Handle sending a query to the AI tool recommender."""
        serializer = QuerySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query = serializer.validated_data["query"]
        access_token = serializer.validated_data["access_token"]

        # Retrieve the guest user based on the access token
        try:
            guest_user = GuestUser.objects.get(access_token=access_token)
        except GuestUser.DoesNotExist:
            return Response(
                {"error": "Guest user not found."}, status=status.HTTP_404_NOT_FOUND
            )

        # Check the number of requests made by this user
        if guest_user.no_of_requests >= 3:
            return Response(
                {"error": "Request limit reached. Only 3 requests allowed."},
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        # Increment the request count
        guest_user.no_of_requests += 1
        guest_user.save()

        ai_response = self.send_query_to_ai(query)
        return Response({"ai_response": ai_response}, status=status.HTTP_200_OK)

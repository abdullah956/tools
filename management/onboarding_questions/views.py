"""Views for managing onboarding questions."""

from drf_spectacular.utils import extend_schema
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from configurations.pagination import CustomPagination

from .models import OnboardingQuestion
from .serializers import OnboardingQuestionSerializer


class OnboardingQuestionViewSet(viewsets.ModelViewSet):
    """A viewset for viewing and editing onboarding questions."""

    serializer_class = OnboardingQuestionSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination

    def get_queryset(self):
        """Return a list of all the onboarding questions for the current user."""
        user = self.request.user
        return OnboardingQuestion.objects.filter(user=user)

    @extend_schema(
        summary="Retrieve a list of onboarding questions for the current user.",
        responses={200: OnboardingQuestionSerializer(many=True)},
    )
    def list(self, request, *args, **kwargs):
        """List all onboarding questions for the current user."""
        return super().list(request, *args, **kwargs)

    @extend_schema(
        summary="Create a new onboarding question for the current user.",
        request=OnboardingQuestionSerializer,
        responses={201: OnboardingQuestionSerializer},
    )
    def create(self, request, *args, **kwargs):
        """Create a new onboarding question for the current user."""
        return super().create(request, *args, **kwargs)

    def perform_create(self, serializer):
        """Save the new onboarding question with the current user."""
        serializer.save(user=self.request.user)

    def get_serializer_context(self):
        """Add user's unique_id to serializer context."""
        context = super().get_serializer_context()
        context["user_unique_id"] = self.request.user.unique_id
        return context

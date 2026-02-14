"""URL configuration for the onboarding questions app."""

# make a urls for the onboarding questions
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import OnboardingQuestionViewSet

router = DefaultRouter()
router.register(
    r"onboarding-questions", OnboardingQuestionViewSet, basename="onboarding-questions"
)

urlpatterns = [
    path("", include(router.urls)),
]

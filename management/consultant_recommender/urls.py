"""URL configuration for Consultant Recommender app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    ConsultantRecommendationViewSet,
    ConsultantSearchViewSet,
    MeetingRequestViewSet,
)

router = DefaultRouter()
router.register(r"searches", ConsultantSearchViewSet, basename="consultant-search")
router.register(r"meetings", MeetingRequestViewSet, basename="meeting-request")
router.register(
    r"recommendations",
    ConsultantRecommendationViewSet,
    basename="consultant-recommendation",
)

app_name = "consultant_recommender"

urlpatterns = [
    path("", include(router.urls)),
]

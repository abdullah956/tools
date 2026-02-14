"""URL configuration for the submit resource form app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import SubmitResourceFormViewSet

router = DefaultRouter()
router.register(
    r"submit-resources", SubmitResourceFormViewSet, basename="submit-resources"
)

urlpatterns = [
    path("", include(router.urls)),
]

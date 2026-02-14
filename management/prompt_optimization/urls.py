"""URL configuration for the prompt optimization app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import UserQueryViewSet

router = DefaultRouter()
router.register(r"user-queries", UserQueryViewSet, basename="userquery")

urlpatterns = [
    path("", include(router.urls)),
]

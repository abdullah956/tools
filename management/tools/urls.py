"""This module contains the urls for the tools."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import ToolViewSet

router = DefaultRouter()
router.register(r"", ToolViewSet)

urlpatterns = [
    path("", include(router.urls)),
]

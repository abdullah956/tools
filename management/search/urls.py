"""URL configuration for search functionality."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import SearchUsageViewSet, SearchViewSet

router = DefaultRouter()
router.register(r"search", SearchViewSet, basename="search")
router.register(r"usage", SearchUsageViewSet, basename="search-usage")

urlpatterns = [
    path("", include(router.urls)),
]

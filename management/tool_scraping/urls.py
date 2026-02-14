"""URL configuration for tool scraping."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    CombinedTextViewSet,
    ScrapingJobViewSet,
    SitePageViewSet,
    ToolSiteViewSet,
)

router = DefaultRouter()
router.register(r"jobs", ScrapingJobViewSet, basename="scraping-job")
router.register(r"sites", ToolSiteViewSet, basename="tool-site")
router.register(r"pages", SitePageViewSet, basename="site-page")
router.register(r"combined-text", CombinedTextViewSet, basename="combined-text")

urlpatterns = [
    path("", include(router.urls)),
]

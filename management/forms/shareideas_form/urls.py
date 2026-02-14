"""URL configuration for the share ideas form app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import ShareIdeasFormViewSet

router = DefaultRouter()
router.register(r"share-ideas", ShareIdeasFormViewSet, basename="share-ideas")

urlpatterns = [
    path("", include(router.urls)),
]

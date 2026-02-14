"""URL configuration for the guest user app."""

from django.urls import include, path
from rest_framework import routers

from .views import GuestUserViewSet

router = routers.DefaultRouter()
router.register(r"guest-user", GuestUserViewSet, basename="guest-user")

urlpatterns = [
    path("", include(router.urls)),
]

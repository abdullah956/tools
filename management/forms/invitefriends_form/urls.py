"""URL configuration for the invite friends form app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import InviteFriendsFormViewSet

router = DefaultRouter()
router.register(r"invite-friends", InviteFriendsFormViewSet, basename="invite-friends")

urlpatterns = [
    path("", include(router.urls)),
]

"""URL configuration for the authentication app."""
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import LinkedInAuthViewSet, UserViewSet

router = DefaultRouter()
router.register(r"users", UserViewSet, basename="user")
router.register(r"linkedin", LinkedInAuthViewSet, basename="linkedin")


urlpatterns = [
    path("", include(router.urls)),
    # path("", include("allauth.urls")),
    # path("email-verification/", include(email_verification_urls)),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

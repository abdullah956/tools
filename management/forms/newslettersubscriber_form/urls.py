"""URL configuration for the newsletter subscriber form app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import NewsletterSubscriberFormViewSet

router = DefaultRouter()
router.register(
    r"newsletter-subscribers",
    NewsletterSubscriberFormViewSet,
    basename="newsletter-subscribers",
)

urlpatterns = [
    path("", include(router.urls)),
]

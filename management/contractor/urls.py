"""URLs for the contractor app."""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import ConsultantSearchViewSet

router = DefaultRouter()
router.register(r"search", ConsultantSearchViewSet, basename="consultant-search")

urlpatterns = [
    path("", include(router.urls)),
]

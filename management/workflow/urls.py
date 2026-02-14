"""URL configuration for the workflow app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import WorkflowViewSet

# Create a separate router for WorkflowViewSet
workflow_router = DefaultRouter()
workflow_router.register(r"workflows", WorkflowViewSet, basename="workflows")

# Include both routers in urlpatterns
urlpatterns = [
    path("", include(workflow_router.urls)),
    path(
        "workflows/<str:pk>/get_workflow_data/",
        WorkflowViewSet.as_view({"get": "get_workflow_data"}),
        name="get_workflow_data",
    ),
    path(
        "workflows/search/",
        WorkflowViewSet.as_view({"get": "search"}),
        name="workflow-search",
    ),
]

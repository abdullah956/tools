"""URL configuration for AI Tool Recommender app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    AIToolSearchViewSet,
    BackgroundTaskViewSet,
    DiscoveredToolViewSet,
    ExplainToolView,
    RefinedQueryViewSet,
    ToolComparisonViewSet,
    WorkflowGenerationViewSet,
    WorkflowImplementationGuideViewSet,
)

router = DefaultRouter()
router.register(r"searches", AIToolSearchViewSet, basename="ai-tool-search")
router.register(r"discovered", DiscoveredToolViewSet, basename="discovered-tool")
router.register(r"workflows", WorkflowGenerationViewSet, basename="workflow-generation")
router.register(r"comparisons", ToolComparisonViewSet, basename="tool-comparison")
router.register(r"tasks", BackgroundTaskViewSet, basename="background-task")
router.register(r"explain", ExplainToolView, basename="explain-tool")
router.register(r"refined-queries", RefinedQueryViewSet, basename="refined-query")
router.register(
    r"implementation-guides",
    WorkflowImplementationGuideViewSet,
    basename="implementation-guide",
)

app_name = "ai_tool_recommender"

urlpatterns = [
    path("", include(router.urls)),
]
